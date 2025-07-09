#!/usr/bin/env python3
"""
Healthcare Q&A Tool - Main Application

A comprehensive healthcare Q&A tool for MediInsight Health Solutions
to research intermittent fasting and metabolic disorders.
"""

import json
import sys
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_settings
from src.processing import DocumentProcessor
from src.qa_system import QAEngine
from src.vector_store import ChromaManager

# Initialize console for rich output
console = Console()


def setup_logging():
    """Setup logging configuration."""
    settings = get_settings()
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger
    logger.add(
        "logs/healthcare_qa.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days"
    )


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Healthcare Q&A Tool for MediInsight Health Solutions."""
    setup_logging()
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


@cli.command()
@click.option('--search-term', '-s', required=True, help='PubMed search term')
@click.option('--max-results', '-m', default=50, help='Maximum number of articles to retrieve')
@click.option('--reset-collection', '-r', is_flag=True, help='Reset the vector collection before ingesting')
def ingest(search_term: str, max_results: int, reset_collection: bool):
    """Ingest articles from PubMed into the vector database."""
    console.print(Panel.fit(
        f"[bold blue]Healthcare Q&A Tool - Data Ingestion[/bold blue]\n"
        f"Search Term: {search_term}\n"
        f"Max Results: {max_results}\n"
        f"Reset Collection: {reset_collection}",
        border_style="blue"
    ))
    
    try:
        processor = DocumentProcessor()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing articles...", total=None)
            
            results = processor.search_and_ingest_pipeline(
                search_term=search_term,
                max_results=max_results,
                reset_collection=reset_collection
            )
        
        if results['success']:
            # Display results table
            table = Table(title="Ingestion Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Articles Found in PubMed", str(results['articles_found_in_pubmed']))
            table.add_row("Articles Processed", str(results['total_articles_processed']))
            table.add_row("High-Relevance Articles", str(results['high_relevance_articles']))
            table.add_row("Articles Added to Vector Store", str(results['articles_added_to_vector_store']))
            
            console.print(table)
            console.print("[bold green]✓ Ingestion completed successfully![/bold green]")
        else:
            console.print(f"[bold red]✗ Ingestion failed: {results['error']}[/bold red]")
            
    except Exception as e:
        console.print(f"[bold red]✗ Error during ingestion: {e}[/bold red]")
        logger.error(f"Ingestion error: {e}")


@cli.command()
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--sources', '-s', is_flag=True, default=True, help='Include source information')
@click.option('--max-docs', '-m', default=5, help='Maximum documents to retrieve')
def ask(question: str, sources: bool, max_docs: int):
    """Ask a question and get an AI-powered answer based on research literature."""
    console.print(Panel.fit(
        f"[bold green]Healthcare Q&A Tool - Question Answering[/bold green]\n"
        f"Question: {question}",
        border_style="green"
    ))
    
    try:
        qa_engine = QAEngine()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating answer...", total=None)
            
            response = qa_engine.ask_question(
                question=question,
                n_documents=max_docs,
                include_sources=sources
            )
        
        # Display answer
        console.print(Panel(
            response['answer'],
            title="[bold]Answer[/bold]",
            border_style="green"
        ))
        
        # Display metadata
        metadata_table = Table(title="Response Metadata")
        metadata_table.add_column("Metric", style="cyan")
        metadata_table.add_column("Value", style="yellow")
        
        metadata_table.add_row("Confidence Score", f"{response['confidence']:.2f}")
        metadata_table.add_row("Sources Used", str(response['sources_count']))
        metadata_table.add_row("Model Used", response.get('model_used', 'Unknown'))
        
        console.print(metadata_table)
        
        # Display sources if requested
        if sources and response.get('sources'):
            console.print("\n[bold]Sources:[/bold]")
            for i, source in enumerate(response['sources'], 1):
                source_info = (
                    f"[bold]{i}. {source['title']}[/bold]\n"
                    f"Authors: {source['authors']}\n"
                    f"Journal: {source['journal']} ({source['publication_date']})\n"
                    f"Study Type: {source['study_type']}\n"
                    f"PMID: {source['pmid']}"
                )
                if source.get('doi'):
                    source_info += f"\nDOI: {source['doi']}"
                
                console.print(Panel(source_info, border_style="dim"))
        
        if response.get('error'):
            console.print(f"[yellow]Warning: {response['error']}[/yellow]")
            
    except Exception as e:
        console.print(f"[bold red]✗ Error during Q&A: {e}[/bold red]")
        logger.error(f"Q&A error: {e}")


@cli.command()
@click.option('--topic', '-t', required=True, help='Research topic to summarize')
@click.option('--max-docs', '-m', default=10, help='Maximum documents to analyze')
def summarize(topic: str, max_docs: int):
    """Generate a research summary for a given topic."""
    console.print(Panel.fit(
        f"[bold purple]Healthcare Q&A Tool - Research Summary[/bold purple]\n"
        f"Topic: {topic}",
        border_style="purple"
    ))
    
    try:
        qa_engine = QAEngine()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating research summary...", total=None)
            
            summary = qa_engine.get_research_summary(
                topic=topic,
                max_documents=max_docs
            )
        
        # Display summary
        console.print(Panel(
            summary['summary'],
            title=f"[bold]Research Summary: {topic}[/bold]",
            border_style="purple"
        ))
        
        # Display analysis if available
        if summary.get('analysis'):
            analysis = summary['analysis']
            
            # Study types table
            if analysis.get('study_types'):
                study_table = Table(title="Study Types Distribution")
                study_table.add_column("Study Type", style="cyan")
                study_table.add_column("Count", style="green")
                
                for study_type, count in analysis['study_types'].items():
                    study_table.add_row(study_type.replace('_', ' ').title(), str(count))
                
                console.print(study_table)
            
            # Publication years
            if analysis.get('publication_years'):
                years = sorted(analysis['publication_years'].keys(), reverse=True)
                console.print(f"\n[bold]Publication Years:[/bold] {', '.join(years[:10])}")
            
            # Research focus areas
            if analysis.get('research_focus_areas'):
                focus_areas = list(analysis['research_focus_areas'].keys())
                console.print(f"[bold]Research Focus Areas:[/bold] {', '.join(focus_areas)}")
        
        console.print(f"\n[bold]Documents Analyzed:[/bold] {summary['document_count']}")
        console.print(f"[bold]Confidence Score:[/bold] {summary.get('confidence', 0):.2f}")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error during summarization: {e}[/bold red]")
        logger.error(f"Summarization error: {e}")


@cli.command()
def stats():
    """Display statistics about the vector database collection."""
    console.print(Panel.fit(
        "[bold cyan]Healthcare Q&A Tool - Collection Statistics[/bold cyan]",
        border_style="cyan"
    ))

    try:
        qa_engine = QAEngine()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Gathering statistics...", total=None)

            insights = qa_engine.get_collection_insights()

        if insights.get('error'):
            console.print(f"[bold red]✗ Error getting statistics: {insights['error']}[/bold red]")
            return

        # Basic stats table
        stats_table = Table(title="Collection Overview")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Documents", str(insights.get('total_documents', 0)))
        stats_table.add_row("Unique Journals", str(insights.get('unique_journals', 0)))
        stats_table.add_row("Collection Name", insights.get('collection_name', 'Unknown'))

        console.print(stats_table)

        # Publication years
        pub_years = insights.get('publication_years', [])
        if pub_years:
            console.print(f"\n[bold]Publication Years:[/bold] {', '.join(pub_years[-10:])}")

        # Article types
        article_types = insights.get('article_types', [])
        if article_types:
            console.print(f"[bold]Article Types:[/bold] {', '.join(article_types[:10])}")

        # Document analysis if available
        if insights.get('document_analysis'):
            analysis = insights['document_analysis']

            if analysis.get('study_types'):
                study_table = Table(title="Study Types in Collection")
                study_table.add_column("Study Type", style="cyan")
                study_table.add_column("Count", style="green")

                for study_type, count in sorted(analysis['study_types'].items(),
                                              key=lambda x: x[1], reverse=True):
                    study_table.add_row(study_type.replace('_', ' ').title(), str(count))

                console.print(study_table)

            avg_relevance = analysis.get('average_relevance', 0)
            console.print(f"\n[bold]Average Relevance Score:[/bold] {avg_relevance:.2f}")

    except Exception as e:
        console.print(f"[bold red]✗ Error getting statistics: {e}[/bold red]")
        logger.error(f"Statistics error: {e}")


@cli.command()
@click.option('--output', '-o', default='collection_export.json', help='Output file path')
def export(output: str):
    """Export the vector database collection to a file."""
    console.print(Panel.fit(
        f"[bold yellow]Healthcare Q&A Tool - Export Collection[/bold yellow]\n"
        f"Output File: {output}",
        border_style="yellow"
    ))

    try:
        chroma_manager = ChromaManager()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Exporting collection...", total=None)

            success = chroma_manager.export_collection(output)

        if success:
            console.print(f"[bold green]✓ Collection exported successfully to {output}[/bold green]")
        else:
            console.print(f"[bold red]✗ Export failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]✗ Error during export: {e}[/bold red]")
        logger.error(f"Export error: {e}")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to reset the collection?')
def reset():
    """Reset the vector database collection (delete all documents)."""
    console.print(Panel.fit(
        "[bold red]Healthcare Q&A Tool - Reset Collection[/bold red]",
        border_style="red"
    ))

    try:
        chroma_manager = ChromaManager()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Resetting collection...", total=None)

            success = chroma_manager.reset_collection()

        if success:
            console.print("[bold green]✓ Collection reset successfully[/bold green]")
        else:
            console.print("[bold red]✗ Reset failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]✗ Error during reset: {e}[/bold red]")
        logger.error(f"Reset error: {e}")


@cli.command()
def interactive():
    """Start interactive Q&A session."""
    console.print(Panel.fit(
        "[bold magenta]Healthcare Q&A Tool - Interactive Mode[/bold magenta]\n"
        "Type 'quit' or 'exit' to end the session.",
        border_style="magenta"
    ))

    try:
        qa_engine = QAEngine()

        while True:
            question = console.input("\n[bold blue]Your question:[/bold blue] ")

            if question.lower() in ['quit', 'exit', 'q']:
                console.print("[bold]Goodbye![/bold]")
                break

            if not question.strip():
                continue

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)

                    response = qa_engine.ask_question(question, include_sources=False)

                console.print(Panel(
                    response['answer'],
                    title="[bold]Answer[/bold]",
                    border_style="green"
                ))

                console.print(f"[dim]Confidence: {response['confidence']:.2f} | "
                            f"Sources: {response['sources_count']}[/dim]")

            except Exception as e:
                console.print(f"[bold red]Error: {e}[/bold red]")

    except KeyboardInterrupt:
        console.print("\n[bold]Session ended.[/bold]")
    except Exception as e:
        console.print(f"[bold red]✗ Error in interactive mode: {e}[/bold red]")
        logger.error(f"Interactive mode error: {e}")


if __name__ == "__main__":
    cli()
