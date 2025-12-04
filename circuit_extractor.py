"""
Circuit Diagram Extractor for RAG System
Extracts circuit diagrams from PDFs and converts them to JSON graph chunks
"""

import json
import re
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import io

import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import easyocr
import networkx as nx
import numpy as np
import cv2

logger = logging.getLogger(__name__)

@dataclass
class CircuitComponent:
    """Represents a circuit component"""
    id: str
    type: str  # resistor, capacitor, inductor, transistor, etc.
    label: str
    value: str
    position: Dict[str, float]  # x, y coordinates
    connections: List[str]  # IDs of connected components

@dataclass
class CircuitGraph:
    """Represents a circuit as a graph"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CircuitDiagramExtractor:
    """Extracts circuit diagrams from PDFs and converts to graph representation"""
    
    # Common circuit component patterns
    COMPONENT_PATTERNS = {
        'resistor': [r'R\d+', r'RES\d+', r'[Rr]esistor'],
        'capacitor': [r'C\d+', r'CAP\d+', r'[Cc]apacitor'],
        'inductor': [r'L\d+', r'IND\d+', r'[Ii]nductor'],
        'transistor': [r'Q\d+', r'TR\d+', r'[Tt]ransistor', r'[Bb][Jj][Tt]', r'[Mm][Oo][Ss]'],
        'diode': [r'D\d+', r'[Dd]iode', r'LED'],
        'voltage_source': [r'V\d+', r'[Vv]cc', r'[Vv]dd', r'[Vv]ss', r'[Vv]in', r'[Vv]out'],
        'ground': [r'GND', r'[Gg]round', r'[Vv]ss'],
        'opamp': [r'U\d+', r'[Oo][Pp][Aa][Mm][Pp]', r'[Oo][Pp]'],
        'switch': [r'SW\d+', r'[Ss]witch'],
        'transformer': [r'T\d+', r'[Tt]ransformer'],
    }
    
    def __init__(
        self,
        lang: List[str] = ['en'],
        logger_obj: Optional[logging.Logger] = None,
        output_dir: str = "artifacts",
    ):
        """Initialize the extractor with OCR reader"""
        self.logger = logger_obj or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing OCR reader with languages=%s", lang)
        self.reader = easyocr.Reader(lang, gpu=False)
        self.logger.info("OCR reader ready")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Artifact output directory: %s", self.output_dir)
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[Image.Image, int]]:
        """Extract images from PDF file"""
        images = []
        try:
            self.logger.info("Opening PDF %s", pdf_path)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Get page as image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append((img, page_num))
                self.logger.debug("Rendered page %s to image (%sx%s)", page_num + 1, img.width, img.height)
            doc.close()
        except Exception as e:
            self.logger.exception("Error extracting images from %s: %s", pdf_path, e)
        return images
    
    def extract_text_from_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Extract text and bounding boxes from image using OCR"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        self.logger.debug("Running OCR on image of size %sx%s", image.width, image.height)
        results = self.reader.readtext(img_array)
        self.logger.debug("OCR produced %s raw results", len(results))
        
        # Format results
        extracted_text = []
        for (bbox, text, confidence) in results:
            # Calculate center position
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            extracted_text.append({
                'text': text.strip(),
                'confidence': confidence,
                'bbox': bbox,
                'position': {'x': center_x, 'y': center_y}
            })
        
        return extracted_text

    def detect_circuit_regions(
        self,
        image: Image.Image,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.9,
        min_line_density: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """Detect likely circuit diagram regions within a page image."""
        self.logger.info("Detecting circuit regions in page image (%sx%s)", image.width, image.height)
        img_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Emphasize drawing lines
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        width, height = image.size
        page_area = width * height
        regions = []

        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / page_area if page_area else 0

            if not (min_area_ratio <= area_ratio <= max_area_ratio):
                continue

            roi = edges[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            density = float(np.count_nonzero(roi)) / roi.size
            if density < min_line_density:
                continue

            regions.append(
                {
                    "bbox": (x, y, w, h),
                    "score": float(density),
                    "region_index": idx,
                }
            )

        # Remove nested/overlapping boxes (simple non-max suppression)
        regions = sorted(regions, key=lambda r: r["score"], reverse=True)
        filtered: List[Dict[str, Any]] = []
        for region in regions:
            x, y, w, h = region["bbox"]
            keep = True
            for kept in filtered:
                kx, ky, kw, kh = kept["bbox"]
                if (
                    x >= kx
                    and y >= ky
                    and x + w <= kx + kw
                    and y + h <= ky + kh
                ):
                    keep = False
                    break
            if keep:
                region["region_index"] = len(filtered)
                filtered.append(region)

        self.logger.info("Detected %s candidate regions (filtered from %s contours)", len(filtered), len(regions))
        return filtered

    def crop_regions(
        self, image: Image.Image, regions: List[Dict[str, Any]], padding_ratio: float = 0.02
    ) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Crop detected regions with optional padding."""
        width, height = image.size
        crops: List[Tuple[Image.Image, Dict[str, Any]]] = []
        self.logger.info("Cropping %s region(s) with padding ratio %.2f", len(regions) or 1, padding_ratio)

        if not regions:
            # Fallback to full page
            full_meta = {
                "bbox": (0, 0, width, height),
                "score": 1.0,
                "region_index": 0,
                "note": "full_page_fallback",
            }
            return [(image.copy(), full_meta)]

        for region in regions:
            x, y, w, h = region["bbox"]
            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)
            left = max(x - pad_w, 0)
            top = max(y - pad_h, 0)
            right = min(x + w + pad_w, width)
            bottom = min(y + h + pad_h, height)

            crop = image.crop((left, top, right, bottom))
            crop_meta = {
                **region,
                "bbox": (left, top, right - left, bottom - top),
            }
            crops.append((crop, crop_meta))

        self.logger.info("Generated %s cropped sub-image(s)", len(crops))
        return crops

    def prepare_llm_payload(
        self, image: Image.Image, format: str = "PNG", quality: int = 90
    ) -> Dict[str, Any]:
        """Convert an image crop into a payload suitable for multimodal LLM APIs."""
        buffer = io.BytesIO()
        save_kwargs = {"format": format}
        if format.upper() in {"JPEG", "JPG"}:
            save_kwargs["quality"] = quality
        image.save(buffer, **save_kwargs)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        self.logger.debug(
            "Prepared LLM payload (%s, %s bytes)",
            save_kwargs["format"],
            buffer.tell(),
        )
        return {
            "type": "input_image",
            "media_type": f"image/{format.lower()}",
            "image_base64": encoded,
        }

    def get_page_output_dir(self, pdf_path: str, page_id: int) -> Path:
        pdf_stem = Path(pdf_path).stem
        page_dir = self.output_dir / pdf_stem / f"page_{page_id:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir

    def annotate_page_image(
        self,
        image: Image.Image,
        regions: List[Dict[str, Any]],
        label_color: Tuple[int, int, int] = (255, 0, 0),
        box_width: int = 5,
    ) -> Image.Image:
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        regions_to_draw = regions or [
            {
                "bbox": (0, 0, image.width, image.height),
                "region_index": 0,
                "score": 1.0,
                "note": "full_page_fallback",
            }
        ]
        for region in regions_to_draw:
            x, y, w, h = region["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=label_color, width=box_width)
            label = f"R{region.get('region_index', 0)}:{region.get('score', 0):.2f}"
            text_pos = (x + 4, max(y - 14, 0))
            draw.text(text_pos, label, fill=label_color)
        return annotated

    def save_annotated_page(
        self,
        image: Image.Image,
        regions: List[Dict[str, Any]],
        page_dir: Path,
        filename: str,
    ) -> Path:
        annotated = self.annotate_page_image(image, regions)
        target_path = page_dir / filename
        annotated.save(target_path)
        self.logger.info("Saved annotated page image to %s", target_path)
        return target_path
    
    def identify_component_type(self, label: str) -> str:
        """Identify component type from label"""
        label_upper = label.upper()
        
        for comp_type, patterns in self.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, label_upper):
                    return comp_type
        
        return 'unknown'
    
    def extract_value(self, text: str, component_type: str) -> str:
        """Extract component value from text"""
        # Look for common value patterns
        value_patterns = [
            r'(\d+\.?\d*)\s*[kKmM]?\s*[ΩOhm]',  # Resistance
            r'(\d+\.?\d*)\s*[uUnNpPmM]?[Ff]',   # Capacitance
            r'(\d+\.?\d*)\s*[uUnNpPmM]?[Hh]',   # Inductance
            r'(\d+\.?\d*)\s*[Vv]',              # Voltage
            r'(\d+\.?\d*)\s*[Aa]',              # Current
        ]
        
        for pattern in value_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ''
    
    def build_graph_from_text(self, extracted_text: List[Dict[str, Any]], 
                             image_size: Tuple[int, int]) -> CircuitGraph:
        """Build circuit graph from extracted text"""
        self.logger.info("Building graph from %s OCR items", len(extracted_text))
        G = nx.Graph()
        components = []
        
        # Process each text element
        for idx, item in enumerate(extracted_text):
            text = item['text']
            position = item['position']
            confidence = item['confidence']
            
            # Skip low confidence or very short text
            if confidence < 0.3 or len(text) < 1:
                continue
            
            # Identify component
            comp_type = self.identify_component_type(text)
            value = self.extract_value(text, comp_type)
            
            # Create component ID
            comp_id = f"comp_{idx}_{text.replace(' ', '_')}"
            
            # Normalize position (0-1 range)
            width, height = image_size
            normalized_pos = {
                'x': position['x'] / width if width > 0 else 0,
                'y': position['y'] / height if height > 0 else 0
            }
            
            component = {
                'id': comp_id,
                'type': comp_type,
                'label': text,
                'value': value,
                'position': normalized_pos,
                'confidence': confidence
            }
            
            components.append(component)
            G.add_node(comp_id, **component)
        
        # Try to infer connections based on proximity and component types
        edges = []
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                # Calculate distance
                pos1 = comp1['position']
                pos2 = comp2['position']
                distance = ((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)**0.5
                
                # Connect if close enough (threshold: 0.1 in normalized coordinates)
                if distance < 0.15:
                    edge = {
                        'source': comp1['id'],
                        'target': comp2['id'],
                        'type': 'connection',
                        'distance': distance
                    }
                    edges.append(edge)
                    G.add_edge(comp1['id'], comp2['id'], **edge)
        
        # Convert to graph structure
        nodes = [comp for comp in components]
        
        graph = CircuitGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                'num_components': len(nodes),
                'num_connections': len(edges),
                'image_size': image_size
            }
        )
        self.logger.info(
            "Graph built with %s component(s) and %s inferred connection(s)",
            len(nodes),
            len(edges),
        )
        return graph
    
    def process_pdf(self, pdf_path: str) -> List[CircuitGraph]:
        """Process a PDF file and extract circuit graphs"""
        self.logger.info("Processing PDF: %s", pdf_path)
        graphs = []
        
        # Extract images from PDF
        images = self.extract_images_from_pdf(pdf_path)
        self.logger.info("Found %s page(s) to analyze", len(images))
        
        for img, page_num in images:
            page_id = page_num + 1
            self.logger.info("Processing page %s", page_id)

            regions = self.detect_circuit_regions(img)
            crops = self.crop_regions(img, regions)
            self.logger.info("Page %s yielded %s region(s)", page_id, len(crops))
            page_dir = self.get_page_output_dir(pdf_path, page_id)
            annotated_filename = f"{Path(pdf_path).stem}_page{page_id:03d}_annotated.png"
            annotated_path = self.save_annotated_page(img, regions, page_dir, annotated_filename)
            
            for crop_img, region_meta in crops:
                region_filename = f"region_{region_meta.get('region_index', 0):02d}.png"
                region_path = page_dir / region_filename
                crop_img.save(region_path)
                self.logger.info(
                    "Saved page %s region %s crop to %s",
                    page_id,
                    region_meta.get('region_index', 0),
                    region_path,
                )

                # Extract text from image
                extracted_text = self.extract_text_from_image(crop_img)
                self.logger.info(
                    "Page %s region %s: %s text elements",
                    page_id,
                    region_meta['region_index'],
                    len(extracted_text),
                )
                
                # Build graph
                graph = self.build_graph_from_text(extracted_text, crop_img.size)
                default_bbox = (0, 0, crop_img.size[0], crop_img.size[1])
                graph.metadata.update({
                    'pdf_path': pdf_path,
                    'page': page_id,
                    'region_index': region_meta.get('region_index', 0),
                    'region_bbox': list(region_meta.get('bbox', default_bbox)),
                    'region_score': region_meta.get('score', 1.0),
                    'num_regions_in_page': len(crops),
                    'region_image_path': str(region_path),
                    'annotated_page_path': str(annotated_path),
                })
                graph.metadata['llm_payload'] = self.prepare_llm_payload(crop_img)
                
                graphs.append(graph)
                self.logger.info(
                    "Page %s region %s captured for downstream LLM (bbox=%s)",
                    page_id,
                    region_meta['region_index'],
                    graph.metadata['region_bbox'],
                )
        
        return graphs
    
    def graph_to_json_chunk(self, graph: CircuitGraph, chunk_id: str) -> Dict[str, Any]:
        """Convert graph to JSON chunk format for RAG"""
        self.logger.debug("Serializing graph %s into chunk", chunk_id)
        # Build text description
        components_desc = []
        for node in graph.nodes:
            desc = f"{node['label']} ({node['type']})"
            if node['value']:
                desc += f" with value {node['value']}"
            components_desc.append(desc)
        
        connections_desc = []
        for edge in graph.edges:
            source_label = next(n['label'] for n in graph.nodes if n['id'] == edge['source'])
            target_label = next(n['label'] for n in graph.nodes if n['id'] == edge['target'])
            connections_desc.append(f"{source_label} connected to {target_label}")
        
        text_content = f"Circuit Diagram:\n"
        text_content += f"Components: {', '.join(components_desc)}\n"
        text_content += f"Connections: {', '.join(connections_desc)}"
        
        return {
            'chunk_id': chunk_id,
            'text': text_content,
            'graph': {
                'nodes': graph.nodes,
                'edges': graph.edges
            },
            'metadata': graph.metadata
        }
    
    def process_folder(self, folder_path: str = '.') -> List[Dict[str, Any]]:
        """Process all PDFs in a folder"""
        folder = Path(folder_path)
        pdf_files = list(folder.glob('*.pdf'))
        
        if not pdf_files:
            self.logger.warning("No PDF files found in %s", folder_path)
            return []
        
        self.logger.info("Found %s PDF file(s) in %s", len(pdf_files), folder_path)
        
        all_chunks = []
        chunk_counter = 0
        
        for pdf_file in pdf_files:
            try:
                graphs = self.process_pdf(str(pdf_file))
                
                for graph in graphs:
                    chunk_id = f"circuit_{chunk_counter}_{pdf_file.stem}_page{graph.metadata['page']}"
                    chunk = self.graph_to_json_chunk(graph, chunk_id)
                    all_chunks.append(chunk)
                    chunk_counter += 1
                    
            except Exception as e:
                self.logger.exception("Error processing %s: %s", pdf_file, e)
                continue
        
        return all_chunks


def main():
    """Main function to extract circuits and save as JSON"""
    import argparse
    
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    parser = argparse.ArgumentParser(description='Extract circuit diagrams from PDFs')
    parser.add_argument('--folder', type=str, default='.', 
                       help='Folder containing PDF files (default: current folder)')
    parser.add_argument('--output', type=str, default='circuit_chunks.json',
                       help='Output JSON file (default: circuit_chunks.json)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = CircuitDiagramExtractor()
    
    # Process all PDFs
    chunks = extractor.process_folder(args.folder)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ Successfully extracted %s circuit diagram chunk(s)", len(chunks))
    logger.info("✓ Saved to %s", output_path)
    
    # Print summary
    logger.info("Summary of extracted chunks:")
    for chunk in chunks:
        metadata = chunk['metadata']
        logger.info(
            "  - %s (page %s region %s): %s components, %s connections",
            metadata.get('pdf_path', 'unknown'),
            metadata.get('page', '?'),
            metadata.get('region_index', 0),
            metadata.get('num_components', 0),
            metadata.get('num_connections', 0),
        )


if __name__ == '__main__':
    main()

