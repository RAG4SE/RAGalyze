"""
Decision module for agent behavior mapping.

This module implements a Decision class that maps from previous context
(prompt, results of previous behaviors) to next prompts using Adalflow's
ConversationMemory for state management.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from adalflow.components.memory import ConversationMemory
from adalflow.core.db import LocalDB


@dataclass
class DecisionContext:
    """Context information for making decisions."""
    prompt: str
    previous_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None


@dataclass
class Decision:
    """A decision that maps context to next prompt."""
    context: DecisionContext
    next_prompt: str
    confidence: float = 1.0
    reasoning: Optional[str] = None


class DecisionMaker:
    """
    A decision-making system that uses Adalflow's ConversationMemory to
    map from previous context to next prompts.

    The decision can be abstracted as:
    (prompt, result of previous behavior, etc) -> (next prompt)
    """

    def __init__(self, memory: Optional[ConversationMemory] = None, user_id: Optional[str] = None):
        """
        Initialize the DecisionMaker.

        Args:
            memory: Optional ConversationMemory instance for storing decision history.
                   If None, a new one will be created.
            user_id: Optional user identifier for the memory system.
        """
        if memory is None:
            self.memory = ConversationMemory(user_id=user_id)
        else:
            self.memory = memory

        self.decision_history: List[Decision] = []

    def make_decision(self,
                     current_prompt: str,
                     previous_result: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Decision:
        """
        Make a decision based on current context and previous results.

        Args:
            current_prompt: The current prompt or question
            previous_result: Result from the previous behavior/action
            metadata: Additional context information

        Returns:
            Decision object containing the next prompt and context
        """
        import time

        # Create context
        context = DecisionContext(
            prompt=current_prompt,
            previous_result=previous_result,
            metadata=metadata or {},
            timestamp=time.time()
        )

        # Store the current prompt in memory
        self.memory.add_system_prompt(current_prompt)

        # If there's a previous result, store it as assistant response
        if previous_result is not None:
            self.memory.add_assistant_response(str(previous_result))

        # Generate next prompt based on context
        next_prompt = self._generate_next_prompt(context)

        # Create decision
        decision = Decision(
            context=context,
            next_prompt=next_prompt,
            confidence=self._calculate_confidence(context),
            reasoning=self._generate_reasoning(context)
        )

        # Store decision in history
        self.decision_history.append(decision)

        return decision

    def _generate_next_prompt(self, context: DecisionContext) -> str:
        """
        Generate the next prompt based on the current context.

        This is a simple implementation. In practice, this could use
        more sophisticated reasoning, LLM calls, or rule-based systems.
        """
        if context.previous_result is None:
            # First interaction, return the original prompt
            return context.prompt

        # Simple pattern: if previous result suggests success, ask for next step
        # If previous result suggests failure, ask for clarification
        result_str = str(context.previous_result).lower()

        if any(keyword in result_str for keyword in ['success', 'complete', 'done', 'finished']):
            return f"Based on the successful completion: {context.prompt[:100]}... What should be the next step?"

        elif any(keyword in result_str for keyword in ['error', 'fail', 'mistake', 'incorrect']):
            return f"The previous attempt failed: {context.prompt}. How should we approach this differently?"

        else:
            return f"Continuing from: {context.prompt[:100]}... Given the result: {str(context.previous_result)[:100]}... What's the next action?"

    def _calculate_confidence(self, context: DecisionContext) -> float:
        """
        Calculate confidence score for the decision.

        Returns:
            Float between 0.0 and 1.0
        """
        # Simple heuristic: more context = higher confidence
        confidence = 0.5  # base confidence

        if context.previous_result is not None:
            confidence += 0.2

        if context.metadata:
            confidence += 0.1

        # Increase confidence based on conversation history
        conversation_length = len(self.memory.get_conversation_history())
        if conversation_length > 0:
            confidence += min(0.2, conversation_length * 0.05)

        return min(1.0, confidence)

    def _generate_reasoning(self, context: DecisionContext) -> str:
        """
        Generate reasoning for the decision.
        """
        if context.previous_result is None:
            return "Initial prompt, no previous context available."

        return f"Decision based on prompt: '{context.prompt[:50]}...' and previous result: '{str(context.previous_result)[:50]}...'"

    def get_decision_history(self) -> List[Decision]:
        """Get the history of all decisions made."""
        return self.decision_history.copy()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history from memory."""
        return self.memory.get_conversation_history()

    def clear_history(self):
        """Clear decision and conversation history."""
        self.decision_history.clear()
        self.memory.clear_conversation()

    def save_to_db(self, db_path: str):
        """Save the memory state to a database file."""
        db = LocalDB(db_path)
        self.memory.save_to_db(db)

    def load_from_db(self, db_path: str):
        """Load memory state from a database file."""
        db = LocalDB(db_path)
        self.memory.load_from_db(db)


# Example usage and testing
def example_usage():
    """Example of how to use the DecisionMaker class."""

    # Create a decision maker
    decision_maker = DecisionMaker(user_id="example_user")

    # First decision (no previous result)
    decision1 = decision_maker.make_decision(
        current_prompt="Write a Python function to calculate fibonacci numbers"
    )
    print(f"Decision 1 - Next prompt: {decision1.next_prompt}")
    print(f"Confidence: {decision1.confidence:.2f}")

    # Second decision (with previous result)
    previous_result = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    decision2 = decision_maker.make_decision(
        current_prompt="Test the fibonacci function",
        previous_result=previous_result,
        metadata={"test_cases": [5, 10, 15]}
    )
    print(f"\nDecision 2 - Next prompt: {decision2.next_prompt}")
    print(f"Confidence: {decision2.confidence:.2f}")
    print(f"Reasoning: {decision2.reasoning}")

    # Show conversation history
    print("\nConversation History:")
    for turn in decision_maker.get_conversation_history():
        print(f"- {turn.get('role', 'unknown')}: {str(turn.get('content', ''))[:100]}...")


if __name__ == "__main__":
    example_usage()