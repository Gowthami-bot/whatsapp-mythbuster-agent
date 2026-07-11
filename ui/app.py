"""
Streamlit UI for WhatsApp Myth Buster Agent.
Provides a chat-style interface for pasting WhatsApp messages
and displays per-claim verdicts with reasoning trail.
"""

import streamlit as st
import os

# Bridge Streamlit secrets into environment variables so os.getenv() picks them up
for key, value in st.secrets.items():
    os.environ[key] = str(value)
import logging
import sys


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import html
import textwrap
from main import run_myth_buster, initialize

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Myth Buster",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Verdict styling ────────────────────────────────────────────────────────────
VERDICT_STYLES = {
    "True": {"emoji": "✅", "color": "#1a7a4a", "bg": "#d4edda"},
    "False": {"emoji": "❌", "color": "#8b1a1a", "bg": "#f8d7da"},
    "Misleading": {"emoji": "⚠️", "color": "#856404", "bg": "#fff3cd"},
    "Unverifiable": {"emoji": "❓", "color": "#495057", "bg": "#e2e3e5"},
}

# ── Sample messages ────────────────────────────────────────────────────────────
SAMPLES = [
    "Drinking hot water with lemon cures cancer! WHO confirmed this. Share immediately!",
    "BREAKING: Government announces free electricity for all households from next month!",
    "Onions kept in room absorb all viruses and bacteria. Place them in every room!",
]


# ── Startup initialization ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Loading Myth Buster Agent...")
def initialize_app():
    """Initialize app resources once per session."""
    try:
        initialize(log_level="INFO")
        return True
    except RuntimeError as e:
        st.error(f"❌ Startup failed: {e}")
        st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render sidebar with instructions and sample messages."""
    with st.sidebar:
        st.title("🔍 Myth Buster")
        st.caption("WhatsApp Fact-Checker Agent")
        st.divider()

        st.markdown("### How it works")
        st.markdown("""
1. Paste any WhatsApp forward
2. Agent extracts individual claims
3. Each claim is searched in our fact-check database
4. If not found, live web search is triggered
5. Evidence is reranked and graded
6. Verdict is generated with confidence score
        """)

        st.divider()
        st.markdown("### Sample Messages")

        for i, sample in enumerate(SAMPLES):
            if st.button(
                f"Sample {i + 1}",
                key=f"sample_{i}",
                use_container_width=True,
            ):
                # Write directly to the text_area's session state key
                st.session_state["message_input"] = sample

        st.divider()
        st.caption("Built with LangGraph + Qdrant + BGE")

def clear_input() -> None:
    """Callback to reset the message input field before rerender."""
    st.session_state["message_input"] = ""
    
# ── Verdict card ───────────────────────────────────────────────────────────────
def render_verdict_card(verdict: dict) -> None:
    """Render a single verdict as a styled HTML card.

    Text fields are HTML-escaped before being embedded so that any
    stray angle brackets in claim/explanation text can't break the markup
    or get displayed as literal tags, and the card itself renders as
    actual HTML (not raw text) via unsafe_allow_html=True.
    """
    claim = html.escape(str(verdict.get("claim", "")))
    verdict_label = str(verdict.get("verdict", "Unverifiable"))
    confidence = verdict.get("confidence", 0)
    explanation = html.escape(str(verdict.get("explanation", "")))

    style = VERDICT_STYLES.get(verdict_label, VERDICT_STYLES["Unverifiable"])

    card_html = textwrap.dedent(f"""
        <div style="
            background-color:{style['bg']};
            border-left:6px solid {style['color']};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:16px;
        ">
            <div style="font-size:1.05rem; font-weight:600; color:#111; margin-bottom:8px;">
                {claim}
            </div>
            <div style="font-size:0.95rem; margin-bottom:6px;">
                <span style="
                    display:inline-block;
                    padding:2px 10px;
                    border-radius:12px;
                    background-color:{style['color']};
                    color:#fff;
                    font-weight:600;
                    margin-right:8px;
                ">{style['emoji']} {html.escape(verdict_label)}</span>
                <span style="color:{style['color']}; font-weight:600;">
                    Confidence: {confidence}%
                </span>
            </div>
            <div style="color:#333; font-size:0.92rem;">
                {explanation}
            </div>
        </div>
    """).strip()

    st.markdown(card_html, unsafe_allow_html=True)

    sources = verdict.get("sources", [])
    if sources:
        with st.expander("📎 Sources", expanded=False):
            for src in sources:
                st.markdown(f"- {src}")


# ── Reasoning trail ────────────────────────────────────────────────────────────
def render_reasoning_trail(trail: list[str]):
    """Render the agent reasoning trail."""
    with st.expander("🧠 Agent Reasoning Trail", expanded=False):
        for i, step in enumerate(trail):
            st.markdown(f"`Step {i + 1}` {step}")


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    """Main Streamlit app entry point."""
    initialize_app()
    render_sidebar()

    st.title("🔍 WhatsApp Myth Buster")
    st.caption("Paste any WhatsApp forward to fact-check it instantly using AI.")
    st.divider()

    # text_area reads/writes directly via its key in session_state
    message = st.text_area(
        label="Paste WhatsApp message here",
        height=150,
        placeholder=(
            "Example: Drinking hot water with lemon cures cancer! "
            "WHO has confirmed this. Share with everyone!"
        ),
        key="message_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_clicked = st.button(
            "🔍 Analyze",
            type="primary",
            use_container_width=True,
        )
    with col2:
        st.button("🗑️ Clear", use_container_width=False, on_click=clear_input)

    if analyze_clicked:
        if not message or not message.strip():
            st.warning("Please paste a WhatsApp message first.")
            return

        with st.spinner("🤖 Agent is analyzing claims..."):
            result = run_myth_buster(
                message=message.strip(),
                thread_id="streamlit_session",
            )

        verdicts = result.get("verdicts", [])
        reasoning_trail = result.get("reasoning_trail", [])

        if not verdicts:
            st.error("No verifiable claims found in this message.")
            if reasoning_trail:
                render_reasoning_trail(reasoning_trail)
            return

        st.divider()
        st.markdown(f"### 📊 Results — {len(verdicts)} claim(s) analyzed")

        for verdict in verdicts:
            render_verdict_card(verdict)

        st.divider()
        render_reasoning_trail(reasoning_trail)

        false_count = sum(1 for v in verdicts if v.get("verdict") == "False")
        misleading_count = sum(1 for v in verdicts if v.get("verdict") == "Misleading")

        if false_count + misleading_count == len(verdicts):
            st.error(
                f"🚨 This message contains {false_count} false "
                f"and {misleading_count} misleading claim(s). Do not forward!"
            )
        elif false_count + misleading_count > 0:
            st.warning("⚠️ This message contains some problematic claims. Verify before forwarding.")
        else:
            st.success("✅ No false claims detected in this message.")


if __name__ == "__main__":
    main()