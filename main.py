import asyncio
import logging
from chartographer.mindmap_generator import MindmapGenerator, ChartographerConfig

async def main():
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("chartographer.main")

    # Define the path to the input file and output directory
    input_file_path = "./workspace/input/SIE.md"
    output_directory = "./workspace/output/"

    # Configure BookWorm (adjust these settings as needed)
    config = ChartographerConfig(
        api_provider="OLLAMA",  # or another provider you're using
        llm_model="qwen3-coder:30b",
        max_tokens=50048,
        temperature=0.2,
        llm_host="http://100.95.157.120:11434"  # Example host, change as needed
    )

    # Initialize the MindmapGenerator with the configuration
    mindmap_generator = MindmapGenerator(config)

    try:
        # Generate and save the mind map from a file
        result = await mindmap_generator.generate_mindmap_from_file(input_file_path, output_directory)
        logger.info(f"Mindmap generated successfully: {result}")

        # You can also generate a mind map directly from text if you prefer
        # with open(input_file_path, 'r', encoding='utf-8') as file:
        #     content = file.read()
        # result = await mindmap_generator.generate_mindmap_from_text(content, "test_document")
        # logger.info(f"Mindmap generated successfully: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())