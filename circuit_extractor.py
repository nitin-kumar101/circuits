import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try to import PyMuPDF for PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyMuPDF not installed. PDF support disabled. Processing only images.")


class CircuitDiagramDetector:
    """
    Circuit diagram region detector using a block-segmentation + filtering approach.
    """
    def __init__(
        self,
        # segmentation parameters
        min_block_area: int = 1000,
        bin_thresh: int = 200,
        morph_kernel_size: int = 3,

        # drawing detection parameters per block
        min_edge_density: float = 0.003,
        hough_threshold: int = 50,
        min_line_length: int = 20,
        max_line_gap: int = 10,
    ):
        self.min_block_area = min_block_area
        self.bin_thresh = bin_thresh
        self.morph_kernel_size = morph_kernel_size

        self.min_edge_density = min_edge_density
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale, enhance contrast and denoise."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return denoised

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Binarize (invert) to get diagram lines/objects as white against black background."""
        _, binary = cv2.threshold(gray, self.bin_thresh, 255, cv2.THRESH_BINARY_INV)
        return binary

    def segment_blocks(self, binary: np.ndarray) -> list:
        """
        Find connected components (external contours) in binary image,
        return list of bounding boxes (x, y, w, h) for each block.
        """
        # Optionally clean up small noise — a small opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []
        h_img, w_img = binary.shape
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Filter out very small areas (noise, stray marks)
            if area < self.min_block_area:
                continue
            # Optionally filter weird shapes (too narrow)
            if w < 20 or h < 20:
                continue
            blocks.append((x, y, w, h))
        logger.info(f"Segmentation yielded {len(blocks)} block(s)")
        return blocks

    def block_has_drawing(self, roi_gray: np.ndarray) -> bool:
        """
        Check whether a given grayscale region likely contains a circuit diagram,
        via edge density and presence of Hough-detected lines.
        """
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        if edge_density < self.min_edge_density:
            return False
        lines = cv2.HoughLinesP(edges,
                                 rho=1,
                                 theta=np.pi/180,
                                 threshold=self.hough_threshold,
                                 minLineLength=self.min_line_length,
                                 maxLineGap=self.max_line_gap)
        if lines is not None and len(lines) > 0:
            return True
        # Fallback: if edge density is sufficiently high, consider it drawing
        if edge_density > (self.min_edge_density * 3):
            return True
        return False

    def detect_circuit_regions(self, image: np.ndarray) -> list:
        """
        Detect individual circuit regions (blocks) in the image.
        Returns list of region dicts with bbox and optionally additional info.
        """
        denoised = self.preprocess(image)
        binary = self.binarize(denoised)
        blocks = self.segment_blocks(binary)

        regions = []
        for idx, (x, y, w, h) in enumerate(blocks):
            roi_gray = denoised[y:y+h, x:x+w]
            if self.block_has_drawing(roi_gray):
                regions.append({
                    'region_id': idx,
                    'bbox': (x, y, w, h)
                })
            else:
                logger.debug(f"Block {idx} rejected: low drawing content or edges.")
        logger.info(f"After filtering: {len(regions)} region(s) considered circuit diagrams")
        return regions

    def crop_region(self, image: np.ndarray, bbox: tuple, padding: int = 5) -> np.ndarray:
        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        return image[y0:y1, x0:x1]

    def visualize_regions(self, image: np.ndarray, regions: list, color=(0,255,0), thickness=2):
        annotated = image.copy()
        for r in regions:
            x, y, w, h = r['bbox']
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(annotated, f"R{r['region_id']}", (x, max(y-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated


def pdf_to_images(pdf_path: str, dpi: int = 200, page_range=None):
    """
    Convert PDF pages to images (color BGR) using PyMuPDF.
    Returns list of (image_array, page_index).
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support is disabled — install PyMuPDF to enable.")
    images = []
    doc = fitz.open(pdf_path)
    start, end = 0, len(doc)
    if page_range:
        start, end = page_range
        end = min(end, len(doc))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for p in range(start, end):
        page = doc[p]
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append((img, p))
        else:
            logger.warning(f"Failed to render page {p}")
    doc.close()
    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract circuit-diagram regions from images or PDFs")
    parser.add_argument("input", type=str, help="Input image or PDF file path")
    parser.add_argument("--output", type=str, default="output_regions", help="Directory to save results")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering PDF (if input is PDF)")
    args = parser.parse_args()

    detector = CircuitDiagramDetector()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".pdf":
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available — please install PyMuPDF (`pip install PyMuPDF`).")
        pages = pdf_to_images(str(input_path), dpi=args.dpi)
        for img, p in pages:
            regions = detector.detect_circuit_regions(img)
            annotated = detector.visualize_regions(img, regions)
            annotated_path = output_dir / f"{input_path.stem}_page{p+1:03d}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            logger.info(f"Saved annotated page {p+1} to {annotated_path}")
            for r in regions:
                cropped = detector.crop_region(img, r['bbox'])
                region_path = output_dir / f"{input_path.stem}_page{p+1:03d}_region_{r['region_id']:02d}.png"
                cv2.imwrite(str(region_path), cropped)
                logger.info(f"Saved region {r['region_id']} to {region_path}")
    else:
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        regions = detector.detect_circuit_regions(img)
        annotated = detector.visualize_regions(img, regions)
        annotated_path = output_dir / f"{input_path.stem}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)
        logger.info(f"Saved annotated image to {annotated_path}")
        for r in regions:
            cropped = detector.crop_region(img, r['bbox'])
            region_path = output_dir / f"{input_path.stem}_region_{r['region_id']:02d}.png"
            cv2.imwrite(str(region_path), cropped)
            logger.info(f"Saved region {r['region_id']} to {region_path}")

    logger.info("Processing complete.")