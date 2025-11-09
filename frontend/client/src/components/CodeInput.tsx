"use client";

import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";

interface CodeInputProps {
  code: string;
  onChange: (code: string) => void;
  onGenerate: () => void;
  isLoading: boolean;
}

export default function CodeInput({ code, onChange, onGenerate, isLoading }: CodeInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Allow Ctrl/Cmd + Enter to generate comments
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (code.trim() && !isLoading) {
        onGenerate();
      }
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle>Python Code Input</CardTitle>
            <CardDescription>Paste your Python code below to generate comments</CardDescription>
          </div>
          <Button
            onClick={onGenerate}
            disabled={!code.trim() || isLoading}
            aria-label={isLoading ? "Generating comments, please wait" : "Generate comments from code"}
            aria-busy={isLoading}
            size="lg"
          >
            {isLoading ? "Generating..." : "Generate Comments"}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <textarea
            id="code-input"
            value={code}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="def sum(a, b):&#10;    return a + b"
            aria-label="Python code input"
            aria-describedby="code-input-help"
            className="w-full h-64 p-4 border border-input bg-background rounded-md font-mono text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 resize-none"
          />
          <p id="code-input-help" className="sr-only">
            Enter Python code. Press Ctrl+Enter or Cmd+Enter to generate comments.
          </p>
        </div>
        <p className="text-sm text-muted-foreground">
          Tip: Press <kbd className="px-2 py-1 bg-muted rounded-md border">Ctrl</kbd> + <kbd className="px-2 py-1 bg-muted rounded-md border">Enter</kbd> to generate comments
        </p>
      </CardContent>
    </Card>
  );
}

