# jbond

Simple agent framework.

## TODO

- [ ] core: study IBM APIs ([link1](https://github.com/IBM/watson-machine-learning-samples/blob/master/README.md), [link2](https://github.com/IBM/watson-machine-learning-samples/blob/master/cloud/notebooks/python_sdk/deployments/foundation_models/chat/Use%20watsonx%2C%20and%20%60mistral-large%60%20to%20make%20simple%20chat%20conversation%20and%20tool%20calls.ipynb))
- [ ] core: find agentic framework. Candidates: [autogen](https://github.com/microsoft/autogen), [pydanticai](https://github.com/pydantic/pydantic-ai), custom-made solution
- [x] rag: find document parser. Solution: [docling](https://github.com/DS4SD/docling)
- [ ] rag: find vector db. Candidates: [chromadb](https://github.com/chroma-core/chroma), [qdrant](https://github.com/qdrant/qdrant), others see [here](https://superlinked.com/vector-db-comparison)
- [ ] tools: develop GoogleSearchTool, CodeExecutionTool

## Development

### Package management with uv

We use ```uv``` for dependency management. Look [here](https://github.com/astral-sh/uv?tab=readme-ov-file) for installation instructions. Here are the basic commands:

```bash
uv sync # Update virtual environment with latest dependencies
uv add <package_name> # Add a new dependency
```
