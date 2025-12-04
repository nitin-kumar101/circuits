"""
Example usage of the Circuit Diagram Extractor
"""

from circuit_extractor import CircuitDiagramExtractor
import json
import logging

logger = logging.getLogger("example_usage")

def configure_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

def example_single_pdf():
    """Example: Process a single PDF file"""
    configure_logging()
    extractor = CircuitDiagramExtractor()
    
    # Process a single PDF
    pdf_path = 'schematic2.pdf'
    graphs = extractor.process_pdf(pdf_path)
    
    # Convert to JSON chunks
    chunks = []
    for idx, graph in enumerate(graphs):
        chunk_id = f"circuit_{idx}_{pdf_path}_page{graph.metadata['page']}"
        chunk = extractor.graph_to_json_chunk(graph, chunk_id)
        chunks.append(chunk)
        
        payload = graph.metadata.get('llm_payload')
        if payload:
            logger.info(
                "Prepared LLM payload for chunk %s (%s, %s base64 chars)",
                chunk_id,
                payload['media_type'],
                len(payload['image_base64']),
            )
        if meta_path := graph.metadata.get('region_image_path'):
            logger.info("Saved region crop for %s at %s", chunk_id, meta_path)
    
    # Save results
    with open('example_output.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info("Processed %s chunk(s) from %s", len(chunks), pdf_path)
    return chunks

def example_batch_processing():
    """Example: Process all PDFs in current folder"""
    configure_logging()
    extractor = CircuitDiagramExtractor()
    
    # Process all PDFs in current folder
    chunks = extractor.process_folder('.')
    
    # Save results
    with open('batch_output.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    logger.info("Total chunks created: %s", len(chunks))
    
    # Print summary
    for chunk in chunks:
        meta = chunk['metadata']
        logger.info("Chunk: %s", chunk['chunk_id'])
        logger.info("  PDF: %s", meta.get('pdf_path', 'unknown'))
        logger.info("  Page: %s", meta.get('page', '?'))
        logger.info("  Region: %s / %s", meta.get('region_index', 0), meta.get('num_regions_in_page', 1))
        logger.info("  Components: %s", meta.get('num_components', 0))
        logger.info("  Connections: %s", meta.get('num_connections', 0))
        llm_payload = meta.get('llm_payload')
        if llm_payload:
            logger.info("  LLM payload ready (media: %s)", llm_payload['media_type'])
        region_img = meta.get('region_image_path')
        if region_img:
            logger.info("  Region image: %s", region_img)
        annotated = meta.get('annotated_page_path')
        if annotated:
            logger.info("  Annotated page: %s", annotated)
    
    return chunks

if __name__ == '__main__':
    # Uncomment the example you want to run:
    
    # Example 1: Process single PDF
    example_single_pdf()
    
    # Example 2: Process all PDFs in folder
    # example_batch_processing()

