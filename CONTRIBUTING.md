# Contributing to Calorie Predictor

Thank you for your interest in contributing to the Calorie Predictor project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/calorie_predictor.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -r requirements.txt`

## Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Check code formatting
black --check src/ tests/
flake8 src/ tests/
```

## Making Changes

1. Write clear, concise commit messages
2. Add tests for any new functionality
3. Update documentation as needed
4. Ensure all tests pass
5. Follow the existing code style

## Submitting a Pull Request

1. Push your changes to your fork
2. Submit a pull request to the main repository
3. Provide a clear description of your changes
4. Link any relevant issues

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep line length under 88 characters (Black default)

## Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Test edge cases and error conditions

## Questions?

Feel free to open an issue for any questions or concerns!