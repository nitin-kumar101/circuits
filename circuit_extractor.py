import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import fitz
# PDF + layout tools
import pdfplumber
PDF_SUPPORT = True

class CircuitDiagramDetector:
    def __init__(self,
                 min_block_area=1000,
                 bin_thresh=200,
                 morph_kernel_size=3,
                 min_edge_density=0.003,
                 hough_threshold=50,
                 min_line_length=20,
                 max_line_gap=10,

                 # PDF-object check thresholds
                 max_text_density: float = 0.005,      # e.g. 0.5% of block area
                 min_graphic_objects: int = 5
                ):
        self.min_block_area = min_block_area
        self.bin_thresh = bin_thresh
        self.morph_kernel_size = morph_kernel_size

        self.min_edge_density = min_edge_density
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # PDF-object filters
        self.max_text_density = max_text_density
        self.min_graphic_objects = min_graphic_objects

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return denoised

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, self.bin_thresh, 255, cv2.THRESH_BINARY_INV)
        return binary

    def segment_blocks(self, binary: np.ndarray) -> list:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < self.min_block_area or w < 20 or h < 20:
                continue
            blocks.append((x, y, w, h))
        logger.info(f"Segmentation: found {len(blocks)} block(s)")
        return blocks

    def block_has_drawing(self, roi_gray: np.ndarray) -> bool:
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        if edge_density < self.min_edge_density:
            return False
        lines = cv2.HoughLinesP(edges,
                                 rho=1, theta=np.pi/180,
                                 threshold=self.hough_threshold,
                                 minLineLength=self.min_line_length,
                                 maxLineGap=self.max_line_gap)
        if lines is not None and len(lines) > 0:
            return True
        if edge_density > (self.min_edge_density * 3):
            return True
        return False

    def pdfblock_check(self, page, block_bbox) -> bool:
        x, y, w, h = block_bbox
        # pdfplumber uses coordinate origin top-left, need to map accordingly
        # page width/height → coordinate system
        # pdfplumber Page.size gives (width, height)
        page_width, page_height = page.width, page.height
        # Convert our bbox (in pixels) to PDF coordinates?
        # --- to simplify: we assume 1:1 mapping after image render; else you'll need scale mapping ---
        x0, top = x, y
        x1, bottom = x + w, y + h

        # Crop PDF page to block
        try:
            cropped = page.crop((x0, top, x1, bottom))
        except Exception as e:
            logger.debug("PDF-crop error, skipping PDF object check")
            return True  # fallback accept

        # Count objects
        text_chars = len(cropped.chars)
        lines = len(cropped.lines)
        rects = len(cropped.rects)
        curves = len(cropped.curves)
        images = len(cropped.images or [])

        block_area = w*h
        text_density = 0
        if block_area > 0:
            # approximate text area as number of chars (each char bounding-box area not considered here)
            text_density = text_chars / block_area

        # Decide: reject if too much text and too few graphics
        if text_density > self.max_text_density and (lines + rects + curves + images) < self.min_graphic_objects:
            logger.info(f"PDF-block reject: high text_density={text_density:.6f}, few graphics ({lines}+{rects}+{curves}+{images})")
            return False
        return True

    def detect_circuit_regions_from_pdf(self, pdf_path: str, dpi: int = 200, page_range=None):
        out = {}
        with pdfplumber.open(pdf_path) as pdf:
            for pno, page in enumerate(pdf.pages):
                if page_range and (pno < page_range[0] or pno >= page_range[1]):
                    continue

                # render as image
                # optionally: use pdfplumber page.to_image(...) or external rendering
                im = page.to_image(resolution=dpi).original  # PIL → convert to OpenCV as needed
                img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                regions = []
                den = self.preprocess(img)
                binr = self.binarize(den)
                blocks = self.segment_blocks(binr)

                for idx, bbox in enumerate(blocks):
                    x,y,w,h = bbox
                    roi = den[y:y+h, x:x+w]
                    if not self.block_has_drawing(roi):
                        continue
                    if not self.pdfblock_check(page, bbox):
                        continue
                    regions.append({'region_id': idx, 'bbox': bbox})
                out[pno] = regions
        return out

    # ... (rest same as before) ...


    def detect_circuit_regions(self, image: np.ndarray) -> list:
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
                logger.debug(f"Block {idx} rejected (low drawing content).")
        logger.info(f"Detected {len(regions)} region(s) after filtering")
        return regions

    def crop_region(self, image: np.ndarray, bbox: tuple, padding: int = 5) -> np.ndarray:
        x, y, w, h = bbox
        H, W = image.shape[:2]
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(W, x + w + padding)
        y1 = min(H, y + h + padding)
        return image[y0:y1, x0:x1]

    def visualize_regions(self, image: np.ndarray, regions: list, color=(0,255,0), thickness: int = 2):
        img = image.copy()
        for r in regions:
            x, y, w, h = r['bbox']
            cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(img, f"R{r['region_id']}", (x, max(y-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


def pdf_to_images(pdf_path: str, dpi: int = 200, page_range=None):
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
            logger.warning(f"Page {p} render failed.")
    doc.close()
    return images


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract circuit blocks from images or PDFs")
    parser.add_argument("input", type=str, help="Input image or PDF path")
    parser.add_argument("--output", type=str, default="output_blocks", help="Directory to save cropped blocks")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")
    args = parser.parse_args()

    detector = CircuitDiagramDetector()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".pdf":
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install PyMuPDF.")
        pages = pdf_to_images(str(input_path), dpi=args.dpi)
        for img, p in pages:
            regions = detector.detect_circuit_regions(img)
            annotated = detector.visualize_regions(img, regions)
            cv2.imwrite(str(output_dir / f"{input_path.stem}_page{p+1:03d}_annotated.png"), annotated)
            for r in regions:
                crop = detector.crop_region(img, r['bbox'])
                cv2.imwrite(str(output_dir / f"{input_path.stem}_page{p+1:03d}_region_{r['region_id']:02d}.png"), crop)
            logger.info(f"Page {p+1} processed: {len(regions)} block(s) saved")
    else:
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Image load failed: {input_path}")
        regions = detector.detect_circuit_regions(img)
        annotated = detector.visualize_regions(img, regions)
        cv2.imwrite(str(output_dir / f"{input_path.stem}_annotated.png"), annotated)
        for r in regions:
            crop = detector.crop_region(img, r['bbox'])
            cv2.imwrite(str(output_dir / f"{input_path.stem}_region_{r['region_id']:02d}.png"), crop)
        logger.info(f"Image processed: {len(regions)} block(s) saved")