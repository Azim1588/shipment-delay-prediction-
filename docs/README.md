# Documentation

This directory contains the documentation for the Shipment Delay Prediction System.

## Structure

- `api/` - API documentation
- `user_guide/` - User guides and tutorials
- `developer/` - Developer documentation
- `deployment/` - Deployment guides

## Building Documentation

To build the documentation:

```bash
# Install documentation dependencies
pip install -r requirements-dev.txt

# Build the docs
cd docs
make html

# View the documentation
open _build/html/index.html
```

## Documentation Guidelines

1. Use clear, concise language
2. Include code examples where appropriate
3. Add screenshots for UI components
4. Keep documentation up to date with code changes
5. Use consistent formatting and structure

## Contributing to Documentation

When contributing to documentation:

1. Follow the existing style and format
2. Test all code examples
3. Update related documentation when making changes
4. Add new sections as needed
5. Review and proofread before submitting

## Documentation Tools

- **Sphinx**: Main documentation generator
- **MyST-Parser**: Markdown support for Sphinx
- **Read the Docs**: Documentation hosting
- **PlantUML**: Diagrams and flowcharts
