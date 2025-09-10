"""
Test module for the function call chain agent.
"""

import unittest
from unittest.mock import Mock, patch
from ragalyze.agent.main import FunctionCallChainAgent
from ragalyze.agent.function_chain_builder import FunctionCallChain


class TestFunctionCallChainAgent(unittest.TestCase):
    """Test cases for the FunctionCallChainAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_path = "/tmp/test_repo"
        with patch('ragalyze.agent.function_chain_builder.RAG') as mock_rag:
            self.agent = FunctionCallChainAgent(self.repo_path)
    
    def test_init(self):
        """Test initialization of the agent."""
        self.assertIsInstance(self.agent, FunctionCallChainAgent)
        self.assertEqual(self.agent.chain_builder.repo_path, self.repo_path)
    
    def test_analyze_function(self):
        """Test analyzing a single function."""
        # Mock the chain builder response
        with patch.object(self.agent.chain_builder, 'build_bidirectional_chain') as mock_build:
            mock_build.return_value = FunctionCallChain(
                target_function="test_function",
                callers=["caller1", "caller2"],
                callees=["callee1", "callee2"],
                forward_chain=["callee1", "callee2"],
                backward_chain=["caller1", "caller2"]
            )
            
            chain = self.agent.analyze_function("test_function")
            
            # Verify the result
            self.assertIsInstance(chain, FunctionCallChain)
            self.assertEqual(chain.target_function, "test_function")
            self.assertEqual(len(chain.callers), 2)
            self.assertEqual(len(chain.callees), 2)
    
    def test_analyze_multiple_functions(self):
        """Test analyzing multiple functions."""
        # Mock the chain builder response
        with patch.object(self.agent.chain_builder, 'build_bidirectional_chain') as mock_build:
            # Set up mock to return different chains for different calls
            mock_build.side_effect = [
                FunctionCallChain(target_function="func1", callers=[], callees=[]),
                FunctionCallChain(target_function="func2", callers=[], callees=[]),
                FunctionCallChain(target_function="func3", callers=[], callees=[])
            ]
            
            function_names = ["func1", "func2", "func3"]
            chains = self.agent.analyze_multiple_functions(function_names)
            
            # Verify the results
            self.assertEqual(len(chains), 3)
            self.assertEqual(chains[0].target_function, "func1")
            self.assertEqual(chains[1].target_function, "func2")
            self.assertEqual(chains[2].target_function, "func3")


class TestFunctionCallChainBuilder(unittest.TestCase):
    """Test cases for the FunctionCallChainBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.repo_path = "/tmp/test_repo"
        with patch('ragalyze.agent.function_chain_builder.RAG') as mock_rag:
            from ragalyze.agent.function_chain_builder import FunctionCallChainBuilder
            self.builder = FunctionCallChainBuilder(self.repo_path)
    
    def test_build_forward_chain(self):
        """Test building forward call chain."""
        with patch.object(self.builder.rag, 'call') as mock_call:
            # Mock the RAG response with [CALL] prefixed tokens
            mock_response = Mock()
            mock_doc = Mock()
            mock_doc.text = "Some text with [CALL]function1 and [CALL]function2 tokens"
            mock_response.documents = [mock_doc]
            mock_call.return_value = [mock_response]
            
            callees = self.builder.build_forward_chain("test_function")
            
            # Verify the result contains the expected callees
            self.assertIn("function1", callees)
            self.assertIn("function2", callees)
    
    def test_build_backward_chain(self):
        """Test building backward call chain."""
        with patch.object(self.builder.rag, 'call') as mock_call:
            # Mock the RAG response with function definitions
            mock_response = Mock()
            mock_doc = Mock()
            mock_doc.text = "def caller1():
    pass

def caller2():
    pass"
            mock_response.documents = [mock_doc]
            mock_call.return_value = [mock_response]
            
            callers = self.builder.build_backward_chain("test_function")
            
            # Verify the result contains the expected callers
            self.assertIn("caller1", callers)
            self.assertIn("caller2", callers)


if __name__ == "__main__":
    unittest.main()