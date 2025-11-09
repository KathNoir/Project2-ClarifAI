"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Play, Pause, SkipForward, RotateCcw } from "lucide-react";

interface VisualizationStep {
  step: number;
  selected_token: string;
  candidates?: Array<{ token: string; score: number }>;
  current_comment: string;
  algorithm_type: "greedy" | "astar";
}

interface AlgorithmVisualizationProps {
  codeTokens: string[];
  greedySteps?: VisualizationStep[];
  astarSteps?: VisualizationStep[];
  algorithm: "greedy" | "astar" | "both";
}

export default function AlgorithmVisualization({
  codeTokens,
  greedySteps = [],
  astarSteps = [],
  algorithm = "both",
}: AlgorithmVisualizationProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1000); // ms per step

  const steps = algorithm === "greedy" ? greedySteps : algorithm === "astar" ? astarSteps : [];

  useEffect(() => {
    if (isPlaying && currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, speed);
      return () => clearTimeout(timer);
    } else if (isPlaying && currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
  }, [isPlaying, currentStep, steps.length, speed]);

  const handlePlay = () => {
    if (currentStep >= steps.length - 1) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  };

  const handlePause = () => setIsPlaying(false);
  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setIsPlaying(false);
    }
  };
  const handleReset = () => {
    setCurrentStep(0);
    setIsPlaying(false);
  };

  if (steps.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Algorithm Visualization</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            Visualization will appear here when you generate comments. Enable "Show Steps" to see how the algorithm works.
          </p>
        </CardContent>
      </Card>
    );
  }

  const currentData = steps[currentStep];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>
            {algorithm === "greedy" ? "Greedy" : algorithm === "astar" ? "A* Beam Search" : "Algorithm"} Visualization
          </CardTitle>
          <Badge variant="outline">
            Step {currentStep + 1} of {steps.length}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <Button
            variant="outline"
            size="sm"
            onClick={isPlaying ? handlePause : handlePlay}
            disabled={steps.length === 0}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isPlaying ? "Pause" : "Play"}
          </Button>
          <Button variant="outline" size="sm" onClick={handleNext} disabled={currentStep >= steps.length - 1}>
            <SkipForward className="h-4 w-4" />
            Next
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
          <div className="ml-auto flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Speed:</label>
            <input
              type="range"
              min="200"
              max="2000"
              step="200"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-24"
            />
            <span className="text-xs text-muted-foreground">{(2000 - speed + 200) / 200}x</span>
          </div>
        </div>

        {/* Current Step Display */}
        <div className="rounded-lg border bg-muted/50 p-4 space-y-3">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="secondary">Selected Token</Badge>
              <span className="font-mono font-semibold text-lg">{currentData.selected_token || "&lt;START&gt;"}</span>
            </div>
            
            {currentData.candidates && currentData.candidates.length > 0 && (
              <div className="mt-3">
                <p className="text-xs text-muted-foreground mb-2">Candidate Tokens:</p>
                <div className="flex flex-wrap gap-2">
                  {currentData.candidates.slice(0, 10).map((candidate, idx) => (
                    <Badge
                      key={idx}
                      variant={candidate.token === currentData.selected_token ? "default" : "outline"}
                      className="font-mono text-xs"
                    >
                      {candidate.token}: {candidate.score.toFixed(2)}
                    </Badge>
                  ))}
                  {currentData.candidates.length > 10 && (
                    <Badge variant="outline" className="text-xs">
                      +{currentData.candidates.length - 10} more
                    </Badge>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Growing Comment */}
          <div className="border-t pt-3">
            <p className="text-xs text-muted-foreground mb-2">Generated Comment So Far:</p>
            <div className="bg-background rounded-md p-3 border">
              <p className="font-mono text-sm whitespace-pre-wrap">
                {currentData.current_comment || "<START>"}
                <span className="animate-pulse">|</span>
              </p>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Progress</span>
            <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
          </div>
          <div className="w-full bg-muted rounded-full h-2">
            <div
              className="bg-primary h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Algorithm Explanation */}
        <div className="rounded-lg bg-blue-50 dark:bg-blue-950/20 p-3 border border-blue-200 dark:border-blue-800">
          <p className="text-xs font-semibold mb-1">How {algorithm === "greedy" ? "Greedy" : "A*"} Works:</p>
          <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
            {algorithm === "greedy" ? (
              <>
                <li>At each step, selects the token with the highest probability</li>
                <li>No lookahead - makes locally optimal choices</li>
                <li>Fast but may miss better global solutions</li>
              </>
            ) : (
              <>
                <li>Maintains multiple candidate paths (beam)</li>
                <li>Uses heuristics to evaluate path quality</li>
                <li>Explores multiple possibilities before committing</li>
              </>
            )}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

