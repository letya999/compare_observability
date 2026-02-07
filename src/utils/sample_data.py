"""Sample data loader for quick start and testing.

Provides sample PDFs and queries for immediate testing without user uploads.
"""

import io
from pathlib import Path
from typing import Tuple

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

from src.logger import logger
from src.config import config


# Sample content about LLM observability
SAMPLE_CONTENT = """
# Understanding LLM Observability: A Comprehensive Guide

## Introduction to LLM Observability

LLM observability refers to the practice of monitoring, tracking, and analyzing the behavior of Large Language Models in production environments. As organizations increasingly deploy LLMs in critical applications, understanding their performance, costs, and reliability becomes essential.

## Key Components of Observability

### 1. Tracing and Spans

Tracing provides visibility into the execution flow of LLM applications. Each operation, from prompt processing to response generation, creates a span that captures timing, inputs, outputs, and metadata. Nested spans reveal the hierarchical structure of complex workflows, such as Retrieval-Augmented Generation (RAG) pipelines.

### 2. Token Counting and Cost Tracking

Token usage directly impacts API costs. Effective observability platforms automatically count input and output tokens, calculate costs based on model pricing, and aggregate expenses over time. This enables teams to optimize prompts, select cost-effective models, and set budget alerts.

### 3. Performance Metrics

Latency is critical for user experience. Observability tools measure:
- Time to First Token (TTFT): How quickly the model starts responding
- Total latency: End-to-end response time
- Throughput: Requests processed per second
- P95 and P99 latencies: Tail latency for reliability assessment

### 4. Quality and Evaluation

Beyond performance, observability includes quality metrics:
- Relevance scores for retrieved documents in RAG systems
- Hallucination detection
- Toxicity and safety checks
- Human feedback collection
- A/B testing different prompts or models

## RAG-Specific Observability

Retrieval-Augmented Generation adds complexity to observability:

### Retrieval Metrics
- Number of documents retrieved
- Relevance scores and ranking
- Chunk overlap and deduplication
- Vector database query performance

### Context Quality
- Context utilization: How much of the retrieved context is actually used
- Source attribution: Tracking which documents influenced the response
- Reranking effectiveness

### End-to-End Visibility
RAG pipelines involve multiple steps: query analysis, retrieval, reranking, generation, and sometimes graph extraction. Observability platforms must visualize this entire flow, showing how data flows through each component.

## Streaming Support

Modern LLM applications often stream responses for better UX. Observability platforms must handle:
- Incremental token tracking
- Partial response capture
- Stream interruption detection
- Time to First Token measurement

## Error Handling and Debugging

Production LLM systems encounter various failures:
- API rate limits and timeouts
- Invalid responses or parsing errors
- Context length exceeded
- Model availability issues

Effective observability captures error traces, provides stack traces, and enables quick root cause analysis.

## Security and Compliance

Observability platforms must address:
- PII detection in prompts and responses
- Prompt injection attack detection
- Audit logs for compliance (SOC2, HIPAA)
- Data retention policies

## Comparison Criteria for Observability Platforms

When evaluating observability platforms, consider:

### Setup and Integration
- Time to first trace
- SDK ease of use
- Auto-instrumentation support
- Documentation quality

### Feature Completeness
- Nested and parallel span support
- Streaming compatibility
- Multi-provider LLM support
- Custom metadata

### Production Readiness
- UI responsiveness
- Search and filtering capabilities
- Alerting and notifications
- Dashboard customization

### Cost and Business
- Pricing model (per trace, per token, flat rate)
- Free tier limitations
- Self-hosted options
- Enterprise features

## Best Practices

1. **Instrument Early**: Add observability from the start, not after issues arise
2. **Sample Strategically**: In high-volume systems, sample traces intelligently
3. **Set Alerts**: Define thresholds for latency, cost, and error rates
4. **Review Regularly**: Analyze traces weekly to identify optimization opportunities
5. **Protect Privacy**: Sanitize PII before sending to third-party platforms

## Emerging Trends

### AI-Powered Debugging
New platforms use AI to:
- Automatically summarize failures
- Suggest prompt improvements
- Detect anomalies and outliers
- Cluster similar issues

### Session Replay
Beyond single traces, some tools offer session replay for multi-turn conversations, preserving agent memory and state across interactions.

### Local-First Development
Developers increasingly want offline observability for local testing without cloud dependencies.

## Conclusion

LLM observability is evolving rapidly. The right platform depends on your specific needs: startup teams may prioritize ease of setup, while enterprises focus on security and self-hosting. By understanding the key capabilities and trade-offs, teams can select tools that provide visibility without adding excessive overhead.

## Glossary

- **Span**: A single unit of work in a trace, with start time, end time, and metadata
- **Trace**: A collection of spans representing a complete request flow
- **TTFT**: Time to First Token, measuring response start latency
- **RAG**: Retrieval-Augmented Generation, combining search with LLM generation
- **Reranking**: Re-scoring retrieved documents for improved relevance
- **P95/P99**: 95th and 99th percentile latency, indicating tail performance

## References and Further Reading

1. OpenTelemetry Specification for LLM Observability
2. LangChain Tracing Documentation
3. Arize AI: Observability for ML Systems
4. Weights & Biases: Experiment Tracking Best Practices
5. The Observability Engineering Book (O'Reilly)
"""

SAMPLE_QUERIES = [
    "What is LLM observability?",
    "How do you measure token costs in production?",
    "What are the key metrics for RAG systems?",
    "Explain the difference between tracing and logging",
    "What security concerns exist with LLM observability?",
    "How do you handle streaming responses in observability platforms?",
    "What is Time to First Token and why does it matter?",
    "Compare different approaches to error handling in LLM applications",
]


def generate_sample_pdf(output_path: Path | None = None) -> Path:
    """Generate a sample PDF about LLM observability.
    
    Args:
        output_path: Where to save the PDF. If None, uses default location.
        
    Returns:
        Path to the generated PDF
    """
    if output_path is None:
        output_path = config.pdf_dir / "sample_llm_observability_guide.pdf"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating sample PDF at {output_path}")
    
    # Create PDF
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Justify',
        alignment=TA_JUSTIFY,
        fontSize=11,
        leading=14,
    ))
    
    # Parse and add content
    lines = SAMPLE_CONTENT.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            elements.append(Spacer(1, 0.2*inch))
            continue
            
        if line.startswith('# '):
            # Main title
            text = line[2:]
            elements.append(Paragraph(text, styles['Title']))
            elements.append(Spacer(1, 0.3*inch))
            
        elif line.startswith('## '):
            # Section heading
            text = line[3:]
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(text, styles['Heading1']))
            elements.append(Spacer(1, 0.1*inch))
            
        elif line.startswith('### '):
            # Subsection heading
            text = line[4:]
            elements.append(Paragraph(text, styles['Heading2']))
            elements.append(Spacer(1, 0.05*inch))
            
        elif line.startswith('- '):
            # Bullet point
            text = line[2:]
            elements.append(Paragraph(f"• {text}", styles['Justify']))
            
        else:
            # Regular paragraph
            elements.append(Paragraph(line, styles['Justify']))
            
    # Build PDF
    doc.build(elements)
    
    logger.info(f"Sample PDF generated successfully: {output_path}")
    return output_path


def get_sample_queries() -> list[str]:
    """Get list of sample queries for testing.
    
    Returns:
        List of sample query strings
    """
    return SAMPLE_QUERIES.copy()


def initialize_sample_data(force: bool = False) -> Tuple[Path, list[str]]:
    """Initialize sample data if not already present.
    
    Args:
        force: If True, regenerate even if sample data exists
        
    Returns:
        Tuple of (pdf_path, sample_queries)
    """
    pdf_path = config.pdf_dir / "sample_llm_observability_guide.pdf"
    
    if force or not pdf_path.exists():
        logger.info("Initializing sample data")
        pdf_path = generate_sample_pdf(pdf_path)
    else:
        logger.info(f"Sample data already exists at {pdf_path}")
        
    return pdf_path, get_sample_queries()


def has_sample_data() -> bool:
    """Check if sample data is already initialized.
    
    Returns:
        True if sample PDF exists
    """
    pdf_path = config.pdf_dir / "sample_llm_observability_guide.pdf"
    return pdf_path.exists()
