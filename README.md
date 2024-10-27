# Smarter Food Choices with RAG using OpenVINO

A Retrieval-Augmented Generation (RAG) Assistant for making informed nutritional choices, powered by Intel OpenVINO and LLaMA 3.1 8B Instruct.

## Project Overview

This project combines advanced AI technologies to create an intelligent nutritional assistant that helps users make informed dietary choices. By leveraging image recognition and real-time data generation, users can simply snap a photo of a food label and receive instant, detailed nutritional insights.

## Key Features

- Image Recognition: Scan food labels using your device's camera
- Real-time Analysis: Get instant nutritional information extraction
- Interactive Q&A: Ask questions about ingredients and receive detailed responses
- Optimized Performance: Leveraging Intel's XPU architecture for efficient processing
- User-friendly Interface: Simple and intuitive interaction model

## Technologies Used

### Intel AI PC
- Optimized for AI and machine learning workloads
- Utilizes Intel's XPU architecture
- Enables smooth handling of complex deep learning models

### Intel OpenVINO Toolkit
- Optimizes deep learning model performance
- Accelerates inference on Intel hardware
- Provides hardware-specific optimizations
- Reduces model latency while maintaining accuracy

### OCR with PyTesseract
- Extracts text from nutritional label images
- Processes various label formats and fonts
- Converts visual information into structured data

### LLaMA 3.1 8B Instruct
- Powers natural language understanding and generation
- Provides contextual nutritional insights
- Enables interactive Q&A capabilities

### LangChain Integration
- Connects various system components
- Facilitates seamless data flow between modules
- Enables complex query processing

## Prerequisites

- Intel AI PC or compatible hardware
- Python 3.8+
- Intel OpenVINO Toolkit
- PyTesseract
- LangChain

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-food-choices-rag.git
cd MCBM
```

2. Install dependencies:
- create a virtual environment and activate it
- open the llm-rag-langchain.ipynb jupyter notebook and run the first cell

3. Install PyTesseract:
```bash
# For Ubuntu
sudo apt-get install tesseract-ocr
# For Windows/WSL
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

1. Start the application:
execute the llm-rag-langchain.ipynb all the way until the last cell
2. Take a photo of a nutritional label or load an existing image
3. Ask questions about the nutritional content
4. Receive detailed, contextualized responses

## Example
![example](](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*9iGM-lqDR8OkKP2J0wRo1A.png))
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Ilias Amchichou
- Ishparsh Uprety
- Visharad Kashyap

Created for the Intel Student Ambassador Fall Hackathon 2024

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact the authors.
