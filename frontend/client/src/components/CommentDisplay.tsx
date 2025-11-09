"use client";

import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

interface CommentResult {
  greedy_comment: string;
  astar_comment: string;
  greedy_runtime_ms: number;
  astar_runtime_ms: number;
  code_tokens: string[];
  greedy_loaded: boolean;
  astar_loaded: boolean;
}

interface CommentDisplayProps {
  result: CommentResult | null;
  error: string | null;
}

export default function CommentDisplay({ result, error }: CommentDisplayProps) {
  if (error) {
    return (
      <Alert variant="destructive" className="mt-6">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!result) {
    return null;
  }

  return (
    <section className="mt-8 space-y-6" aria-label="Generated comments">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Greedy Comment */}
        <Card aria-labelledby="greedy-title">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle id="greedy-title" className="flex items-center gap-2">
                Greedy Algorithm
                {!result.greedy_loaded && (
                  <Badge variant="destructive" aria-label="Model not loaded">Not Loaded</Badge>
                )}
              </CardTitle>
              <Badge variant="outline" className="text-xs">
                {result.greedy_runtime_ms > 0 ? `${result.greedy_runtime_ms.toFixed(2)} ms` : "N/A"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="rounded-md bg-muted p-4 min-h-[100px]">
              <p 
                className={`whitespace-pre-wrap ${
                  result.greedy_comment.startsWith("[") ? "italic text-muted-foreground" : "text-foreground"
                }`}
                aria-label="Generated comment from greedy algorithm"
              >
                {result.greedy_comment || "[Greedy model not loaded]"}
              </p>
            </div>
          </CardContent>
          <CardFooter className="text-sm text-muted-foreground">
            Fast • ~0.19ms average
          </CardFooter>
        </Card>

        {/* A* Comment */}
        <Card aria-labelledby="astar-title">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle id="astar-title" className="flex items-center gap-2">
                A* Algorithm
                {!result.astar_loaded && (
                  <Badge variant="destructive" aria-label="Model not loaded">Not Loaded</Badge>
                )}
              </CardTitle>
              <Badge variant="outline" className="text-xs">
                {result.astar_runtime_ms > 0 ? `${result.astar_runtime_ms.toFixed(2)} ms` : "N/A"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="rounded-md bg-muted p-4 min-h-[100px]">
              <p 
                className={`whitespace-pre-wrap ${
                  result.astar_comment.startsWith("[") ? "italic text-muted-foreground" : "text-foreground"
                }`}
                aria-label="Generated comment from A* algorithm"
              >
                {result.astar_comment || "[A* model not loaded]"}
              </p>
            </div>
          </CardContent>
          <CardFooter className="text-sm text-muted-foreground">
            High Quality • ~13.5ms average
          </CardFooter>
        </Card>
      </div>

      {/* Code Tokens */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Extracted Code Tokens</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {result.code_tokens.length > 0 ? (
              result.code_tokens.map((token, idx) => (
                <Badge key={idx} variant="secondary">
                  {token}
                </Badge>
              ))
            ) : (
              <span className="text-muted-foreground">None</span>
            )}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}

