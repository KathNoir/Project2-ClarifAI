"use client";

import { useState } from "react";
import CodeInput from "@/components/CodeInput";
import CommentDisplay from "@/components/CommentDisplay";
import AlgorithmVisualization from "@/components/AlgorithmVisualization";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface CommentResult {
  greedy_comment: string;
  astar_comment: string;
  greedy_runtime_ms: number;
  astar_runtime_ms: number;
  code_tokens: string[];
  greedy_loaded: boolean;
  astar_loaded: boolean;
  greedy_steps?: Array<{
    step: number;
    selected_token: string;
    candidates?: Array<{ token: string; score: number }>;
    current_comment: string;
    algorithm_type: "greedy" | "astar";
  }>;
  astar_steps?: Array<{
    step: number;
    selected_token: string;
    candidates?: Array<{ token: string; score: number }>;
    current_comment: string;
    algorithm_type: "greedy" | "astar";
  }>;
}

export default function Home() {
  const [code, setCode] = useState("");
  const [result, setResult] = useState<CommentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showVisualization, setShowVisualization] = useState(false);

  const handleGenerate = async () => {
    if (!code.trim()) {
      setError("Please enter some Python code");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Call backend API
            const response = await fetch("http://localhost:8001/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: code,
          max_length: 20,
          enable_visualization: showVisualization,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to generate comments" }));
        throw new Error(errorData.detail || "Failed to generate comments");
      }

      const data: CommentResult = await response.json();
      
      // Show results even if one model isn't loaded (graceful degradation)
      setResult(data);
      
      // Show warning if models not loaded, but don't block the result
      if (!data.greedy_loaded || !data.astar_loaded) {
        console.warn("Some models not loaded:", {
          greedy: data.greedy_loaded,
          astar: data.astar_loaded
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Code Comment Generator
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Compare Greedy vs A* comment generation algorithms
          </p>
        </header>

        {/* Code Input */}
        <section aria-labelledby="code-input-section">
          <h2 id="code-input-section" className="sr-only">
            Code Input Section
          </h2>
          <CodeInput
            code={code}
            onChange={setCode}
            onGenerate={handleGenerate}
            isLoading={isLoading}
          />
        </section>

        {/* Visualization Toggle */}
        <Card className="mt-4 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold mb-1">Algorithm Visualization</h3>
              <p className="text-sm text-muted-foreground">
                Watch how each algorithm generates comments step-by-step
              </p>
            </div>
            <Button
              variant={showVisualization ? "default" : "outline"}
              onClick={() => setShowVisualization(!showVisualization)}
              disabled={isLoading}
            >
              {showVisualization ? "Hide" : "Show"} Visualization
            </Button>
          </div>
        </Card>

        {/* Results */}
        <CommentDisplay result={result} error={error} />

        {/* Algorithm Visualization */}
        {showVisualization && result && (
          <div className="mt-8 space-y-6">
            {result.greedy_steps && result.greedy_steps.length > 0 && (
              <AlgorithmVisualization
                codeTokens={result.code_tokens}
                greedySteps={result.greedy_steps}
                algorithm="greedy"
              />
            )}
            {result.astar_steps && result.astar_steps.length > 0 && (
              <AlgorithmVisualization
                codeTokens={result.code_tokens}
                astarSteps={result.astar_steps}
                algorithm="astar"
              />
            )}
            {(!result.greedy_steps || result.greedy_steps.length === 0) &&
              (!result.astar_steps || result.astar_steps.length === 0) && (
                <Card>
                  <CardContent className="p-6">
                    <p className="text-muted-foreground text-center">
                      Visualization data not available. Make sure models are loaded and enable visualization before generating.
                    </p>
                  </CardContent>
                </Card>
              )}
          </div>
        )}
      </main>
    </div>
  );
}
