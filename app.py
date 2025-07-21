"""
PM Knowledge Assistant - Web Interface
A Gradio-based UI for querying Product Management literature
"""
import gradio as gr
import sys
from pathlib import Path
import logging
from typing import Tuple, List
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import PMKnowledgeRAG
from src.state_manager import StateManager, ProcessingStatus
from src.ui import SourceManager, SelectionHandlers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components (global to maintain state)
rag = None
state_manager = None
source_manager = None
selection_handlers = None

def initialize_rag():
    """Initialize the RAG pipeline and all UI components"""
    global rag, state_manager, source_manager, selection_handlers
    try:
        rag = PMKnowledgeRAG()
        state_manager = StateManager()
        source_manager = SourceManager(state_manager)
        selection_handlers = SelectionHandlers(source_manager)
        return True, "✅ System initialized successfully!"
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {str(e)}")
        return False, f"❌ Initialization failed: {str(e)}"

# Clean, simple wrapper functions using our UI services
def refresh_book_sources():
    """Refresh available book sources"""
    if not selection_handlers:
        return gr.CheckboxGroup(choices=[], value=[]), "System not initialized"
    return selection_handlers.refresh_sources()

def select_all_books():
    """Select all available books"""
    if not selection_handlers:
        return []
    return selection_handlers.select_all_available()

def clear_all_books():
    """Clear all book selections"""
    if not selection_handlers:
        return []
    return selection_handlers.clear_all_selections()

def update_selection_display(selected_books):
    """Update selection summary display"""
    if not selection_handlers:
        return "System not initialized"
    return selection_handlers.update_selection_summary(selected_books)

def format_sources(sources: List[dict]) -> str:
    """Format sources for display"""
    if not sources:
        return "No sources found."
    
    source_text = "📚 **Sources Used:**\n\n"
    for i, source in enumerate(sources, 1):
        source_text += f"{i}. **{source['book_title']}**\n"
        source_text += f"   - Author: {source['author']}\n"
        source_text += f"   - Chapter: {source['chapter']}\n"
        if source.get('section'):
            source_text += f"   - Section: {source['section']}\n"
        source_text += f"   - Relevance: {source['similarity_score']:.2%}\n\n"
    
    return source_text

def answer_question(question: str, selected_books: List[str] = None) -> Tuple[str, str]:
    """Answer a product management question with enhanced book filtering"""
    if not rag or not selection_handlers:
        return "❌ System not initialized. Please refresh the page.", ""
    
    if not question.strip():
        return "Please enter a question.", ""
    
    # Validate selection
    is_valid, validation_msg = selection_handlers.validate_selection(selected_books or [])
    if not is_valid:
        return validation_msg, ""
    
    try:
        # Calculate smart k value based on selection
        k = selection_handlers.calculate_smart_k(selected_books or [])
        
        # Generate answer with book filtering
        result = rag.generate_answer(question, k=k, selected_books=selected_books)
        
        # Format sources
        sources_text = format_sources(result['sources'])
        
        # Add selection info to answer if books were filtered
        if selected_books:
            selected_info = selection_handlers.update_selection_summary(selected_books)
            sources_text = f"{selected_info}\n\n{sources_text}"
        
        return result['answer'], sources_text
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"❌ Error: {str(e)}", ""

def compare_perspectives(topic: str, selected_books: List[str] = None) -> str:
    """Compare different authors' perspectives on a topic with book filtering"""
    if not rag or not selection_handlers:
        return "❌ System not initialized. Please refresh the page."
    
    if not topic.strip():
        return "Please enter a topic to compare perspectives."
    
    # Validate selection
    is_valid, validation_msg = selection_handlers.validate_selection(selected_books or [])
    if not is_valid:
        return validation_msg
    
    try:
        # Calculate smart k value based on selection
        k = selection_handlers.calculate_smart_k(selected_books or [])
        # For perspectives, we want more sources to compare
        k = min(k * 2, 20)  # Double k but cap at 20
        
        result = rag.compare_perspectives(topic, k=k, selected_books=selected_books)
        
        comparison_text = f"**Comparing perspectives on: {topic}**\n\n"
        
        # Add selection info if books were filtered
        if selected_books:
            selected_info = selection_handlers.update_selection_summary(selected_books)
            comparison_text += f"{selected_info}\n\n"
        
        comparison_text += f"Authors analyzed: {', '.join(result['authors'])}\n\n"
        comparison_text += result['comparison']
        comparison_text += f"\n\n---\n📊 *Based on {result['chunks_used']} text segments*"
        
        return comparison_text
        
    except Exception as e:
        logger.error(f"Error comparing perspectives: {str(e)}")
        return f"❌ Error: {str(e)}"

# Example questions
example_questions = [
    "What are the key skills needed for a product manager?",
    "How should product managers handle feature prioritization?",
    "What frameworks are recommended for product roadmapping?",
    "How do you measure product-market fit?",
    "What's the best approach to user research for product managers?",
    "How should PMs work with engineering teams?",
    "What metrics should product managers track?",
    "How do you write effective product requirements?",
    "What's the role of a PM in agile development?",
    "How do you handle stakeholder management as a PM?"
]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="PM Knowledge Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📚 Super PM Knowledge Assistant
        
        Ask questions about Product Management and get evidence-based answers from multiple PM books.
        This system searches through a curated library of PM literature to provide comprehensive, 
        well-sourced answers to your questions.
        """)
        
        # Initialize on load
        init_status = gr.Markdown("🔄 Initializing system...")
        
        with gr.Tab("Ask a Question"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Enter your product management question...",
                        lines=3
                    )
                    
                    book_selection = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="📖 Select Books to Search",
                        info="Choose which books to include in your search. Smart defaults auto-select completed books."
                    )
                    
                    # Selection summary
                    selection_summary = gr.Markdown("📊 Selection: Loading...")
                    
                    with gr.Row():
                        select_all_btn = gr.Button("✅ Select All", size="sm", variant="secondary")
                        select_none_btn = gr.Button("❌ Clear All", size="sm", variant="secondary") 
                        refresh_sources_btn = gr.Button("🔄 Refresh", size="sm", variant="primary")
                    
                    ask_button = gr.Button("🔍 Get Answer", variant="primary")
                    
                    gr.Examples(
                        examples=example_questions,
                        inputs=question_input,
                        label="Example Questions"
                    )
                
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
        
        with gr.Tab("Compare Perspectives"):
            gr.Markdown("""
            ### Compare Different Authors' Perspectives
            Enter a topic to see how different PM authors approach it.
            """)
            
            topic_input = gr.Textbox(
                label="Topic to Compare",
                placeholder="e.g., 'user research', 'prioritization', 'metrics'",
                lines=2
            )
            
            compare_book_selection = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="📖 Select Books to Compare",
                info="Choose which books to include in the comparison. Smart defaults auto-select completed books."
            )
            
            # Selection summary for compare tab
            compare_selection_summary = gr.Markdown("📊 Selection: Loading...")
            
            with gr.Row():
                compare_select_all_btn = gr.Button("✅ Select All", size="sm", variant="secondary")
                compare_select_none_btn = gr.Button("❌ Clear All", size="sm", variant="secondary") 
                compare_refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="primary")
            
            compare_button = gr.Button("🔄 Compare Perspectives", variant="primary")
            
            comparison_output = gr.Markdown(label="Comparison")
            
            gr.Examples(
                examples=[
                    "feature prioritization",
                    "product-market fit",
                    "user research methods",
                    "metrics and KPIs",
                    "stakeholder management"
                ],
                inputs=topic_input,
                label="Example Topics"
            )
        
        with gr.Tab("📚 Available Sources"):
            gr.Markdown("### Book Processing Status")
            gr.Markdown("View the current status of all books in the system.")
            
            # Display the sources info without accordion
            sources_display = gr.Markdown("Loading available sources...")
            
            # Add a refresh button for this tab
            sources_refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            
        with gr.Tab("About"):
            gr.Markdown("""
            ### About PM Knowledge Assistant
            
            This tool provides access to insights from multiple Product Management books including:
            
            - **Cracking the PM Career** by Jackie Bavaro & Gayle Laakmann McDowell
            - **The PM Interview** by Lewis C. Lin
            - **Decode and Conquer** by Lewis C. Lin
            - **The Lean Product Playbook** by Dan Olsen
            - **The Lean Startup** by Eric Ries
            - And more...
            
            #### Features:
            - 🔍 Semantic search across all books
            - 📚 Source attribution for every answer
            - 🔄 Compare different authors' perspectives
            - 💡 Evidence-based insights
            
            #### How it works:
            1. Your question is converted to embeddings
            2. Similar content is retrieved from the book database
            3. An AI synthesizes the information into a comprehensive answer
            4. Sources are clearly cited for verification
            """)
        
        # Enhanced event handlers using our clean UI services
        ask_button.click(
            fn=answer_question,
            inputs=[question_input, book_selection],
            outputs=[answer_output, sources_output]
        )
        
        refresh_sources_btn.click(
            fn=lambda: refresh_book_sources()[0],  # Only return the checkbox choices
            outputs=book_selection
        )
        
        select_all_btn.click(
            fn=select_all_books,
            outputs=book_selection
        )
        
        select_none_btn.click(
            fn=clear_all_books,
            outputs=book_selection
        )
        
        # Update selection summary when books are selected/deselected
        book_selection.change(
            fn=update_selection_display,
            inputs=book_selection,
            outputs=selection_summary
        )
        
        # Event handlers for Compare tab book selection (reusing same functions)
        compare_refresh_btn.click(
            fn=lambda: refresh_book_sources()[0],  # Only return the checkbox choices
            outputs=compare_book_selection
        )
        
        compare_select_all_btn.click(
            fn=select_all_books,
            outputs=compare_book_selection
        )
        
        compare_select_none_btn.click(
            fn=clear_all_books,
            outputs=compare_book_selection
        )
        
        compare_book_selection.change(
            fn=update_selection_display,
            inputs=compare_book_selection,
            outputs=compare_selection_summary
        )
        
        compare_button.click(
            fn=compare_perspectives,
            inputs=[topic_input, compare_book_selection],
            outputs=comparison_output
        )
        
        # Event handler for Available Sources tab
        sources_refresh_btn.click(
            fn=lambda: refresh_book_sources()[1],  # Only return the sources display
            outputs=sources_display
        )
        
        # Initialize system and load sources on startup
        def init_system_and_sources():
            success, message = initialize_rag()
            if success and selection_handlers:
                # Also refresh sources on startup
                book_choices, sources_info_text = refresh_book_sources()
                # Return data for all components
                return (
                    message,
                    book_choices, "📚 Ready to search!",
                    book_choices, "📚 Ready to compare!",
                    sources_info_text  # For the Available Sources tab
                )
            return (
                message,
                gr.CheckboxGroup(choices=[], value=[]), "❌ Not ready",
                gr.CheckboxGroup(choices=[], value=[]), "❌ Not ready",
                "❌ System not initialized"
            )
        
        demo.load(
            fn=init_system_and_sources,
            outputs=[
                init_status,
                book_selection, selection_summary,
                compare_book_selection, compare_selection_summary,
                sources_display
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        "data/faiss_index.bin",
        "data/chunks_metadata.json"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run the following commands first:")
        print("1. python process_all_books.py")
        print("2. python generate_embeddings.py")
        sys.exit(1)
    
    # Initialize and launch
    print("Initializing PM Knowledge Assistant...")
    success, message = initialize_rag()
    
    if not success:
        print(message)
        sys.exit(1)
    
    print(message)
    print("\nLaunching web interface...")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )