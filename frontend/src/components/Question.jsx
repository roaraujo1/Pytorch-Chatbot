import { useState } from "react";
import ReactMarkdown from "react-markdown";
import api from "../api"; 

function Question() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await api.post("/ask", { question });
      setResponse(res.data.answer);
    } catch (err) {
      setResponse("Error: could not reach the chatbot.");
      console.error(err);
    } finally {
      setLoading(false);
    }
    
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit" disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
      {response && (
        <div className="response">
          <ReactMarkdown>{response}</ReactMarkdown> 
        </div>

      )}
    </div>
  );
}
//How do I create a custom Dataset class in PyTorch?
export default Question;