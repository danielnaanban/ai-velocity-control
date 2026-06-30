import { useState, useEffect, useCallback } from "react";
import { 
  Activity, 
  Zap, 
  ShieldCheck, 
  Cpu, 
  TrendingUp, 
  Terminal, 
  BarChart3, 
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  Loader2
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Toaster } from "sonner";
import { useTradingData } from "@/hooks/use-trading-data";

function App() {
  const [logs, setLogs] = useState<string[]>([]);
  const [systemStatus, setSystemStatus] = useState("Idle");
  const [isAutoMode, setIsAutoMode] = useState(false);
  const { loading: isLoading, signal, fetchSignal } = useTradingData();

  useEffect(() => {
    const bootLogs = [
      "[SYSTEM] Initializing AI Velocity Trader...",
      "[MODULE] Neural Analysis Engine: LOADING WEIGHTS",
      "[MODULE] Market Scanning Layer: CONNECTING TO API",
      "[MODULE] Risk Management: CHECKING CIRCUIT BREAKERS",
      "[MODULE] Execution Layer: ASYNC HANDLERS READY",
      "[READY] System Online. 24/7 Scanning Active."
    ];

    let i = 0;
    const interval = setInterval(() => {
      if (i < bootLogs.length) {
        setLogs(prev => [...prev, bootLogs[i]]);
        i++;
      } else {
        clearInterval(interval);
      }
    }, 800);

    return () => clearInterval(interval);
  }, []);

  // Health check on load
  useEffect(() => {
    const API_URL = import.meta.env.VITE_API_URL || '';
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (response.ok) {
          setSystemStatus("Online");
        } else {
          setSystemStatus("Offline");
        }
      } catch (err) {
        setSystemStatus("Offline");
      }
    };
    checkHealth();
  }, []);

  // Fetch real logs every 3 seconds
  useEffect(() => {
    const API_URL = import.meta.env.VITE_API_URL || '';
    const fetchLogs = async () => {
      try {
        const response = await fetch(`${API_URL}/api/logs`);
        if (response.ok) {
          const data = await response.json();
          if (data && data.length > 0) {
            setLogs(prev => {
              const newLogs = data.filter((log: string) => !prev.includes(log));
              if (newLogs.length === 0) return prev;
              return [...prev, ...newLogs].slice(-50);
            });
          }
        }
      } catch (err) {
        console.error("Logs fetch error:", err);
      }
    };

    const interval = setInterval(fetchLogs, 3000);
    return () => clearInterval(interval);
  }, []);

  // Handle polling when Auto Mode is ON
  useEffect(() => {
    if (!isAutoMode) return;

    const pollInterval = setInterval(() => {
      fetchSignal('EURUSD');
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [isAutoMode, fetchSignal]);

  const handleGenerateSignal = useCallback(async () => {
    const data = await fetchSignal('EURUSD');
    if (data && data.action === 'HOLD') {
      toast.error("Signal blocked by Risk Module", {
        description: `Confidence ${(data.confidence * 100).toFixed(1)}% is below threshold.`,
      });
    } else if (data) {
      toast.success(`New signal detected: ${data.action} ${data.pair}`, {
        description: `Confidence: ${(data.confidence * 100).toFixed(1)}%`,
      });
    }
  }, [fetchSignal]);

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-white font-sans selection:bg-emerald-500/30">
      <Toaster position="top-right" theme="dark" />
      
      {/* Hero Section */}
      <div className="relative h-[400px] overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-40 grayscale"
          style={{ backgroundImage: 'url(https://storage.googleapis.com/dala-prod-public-storage/generated-images/f3273c13-429d-488b-a168-810f8bc98b10/ai-velocity-trader-dashboard-background-a6cde243-1782510214729.webp)' }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-[#0a0a0b]" />
        
        <div className="relative container mx-auto px-6 pt-20">
          <div className="flex items-center gap-3 mb-4">
            <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
              <Activity className="w-3 h-3 mr-1 animate-pulse" />
              Live Execution
            </Badge>
            <Badge variant="outline" className="border-zinc-700 text-zinc-400 bg-zinc-900/50">
              Status: {systemStatus}
            </Badge>
          </div>
          <h1 className="text-6xl font-black tracking-tighter mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/50">
            AI VELOCITY <span className="text-emerald-500 italic">TRADER</span>
          </h1>
          <p className="max-w-xl text-lg text-zinc-400 leading-relaxed mb-8">
            Sub-millisecond forex execution powered by a lightweight Transformer-LSTM hybrid. 
            Risk-first architecture optimized for 24/7 high-frequency trading.
          </p>
          <div className="flex gap-4">
            <Button size="lg" variant="outline" className="rounded-none border-zinc-700 hover:bg-zinc-800">
              VIEW BACKTESTS
            </Button>
          </div>
        </div>
      </div>

      {/* Modules Grid */}
      <div className="container mx-auto px-6 -mt-10 mb-20">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-4 bg-zinc-900/80 p-4 border border-zinc-800 backdrop-blur-md w-full md:w-auto">
            <div className="flex items-center space-x-2">
              <Switch 
                id="auto-mode" 
                checked={isAutoMode} 
                onCheckedChange={setIsAutoMode}
              />
              <Label htmlFor="auto-mode" className="text-sm font-bold tracking-tighter cursor-pointer">
                AUTO MODE: <span className={isAutoMode ? "text-emerald-500" : "text-zinc-500"}>{isAutoMode ? "ON" : "OFF"}</span>
              </Label>
            </div>
          </div>

          <div className="fixed bottom-0 left-0 right-0 p-4 bg-black/80 backdrop-blur-lg border-t border-zinc-800 z-50 md:relative md:bg-transparent md:border-none md:p-0 md:z-auto">
            <Button 
              id="btn-generate-signal"
              size="lg" 
              className="w-full md:w-[280px] bg-gradient-to-r from-orange-600 to-orange-400 hover:from-orange-500 hover:to-orange-300 text-white font-bold px-8 rounded-none border-b-4 border-orange-800 disabled:opacity-70 transition-all active:translate-y-1 active:border-b-0 h-14" 
              onClick={handleGenerateSignal}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ANALYZING...
                </>
              ) : (
                "GENERATE AI SIGNAL"
              )}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-xl">
            <CardHeader className="pb-2">
              <Cpu className="w-5 h-5 text-emerald-500 mb-2" />
              <CardTitle className="text-sm font-medium uppercase tracking-widest text-zinc-500">Neural Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {signal?.action === "HOLD" ? (
                <div className="text-2xl font-bold text-orange-500">HOLD</div>
              ) : signal ? (
                <div className="text-2xl font-bold">{(signal.confidence * 100).toFixed(1)}%</div>
              ) : (
                <div className="text-2xl font-bold text-zinc-700">HOLD</div>
              )}
              <p className="text-xs text-zinc-500 mt-1">
                {signal?.action === "HOLD" 
                  ? `Reason: Confidence ${signal.confidence.toFixed(2)} < 0.75 Threshold` 
                  : signal ? "Inference Confidence" : "Awaiting Signal"}
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-xl">
            <CardHeader className="pb-2">
              <BarChart3 className="w-5 h-5 text-emerald-500 mb-2" />
              <CardTitle className="text-sm font-medium uppercase tracking-widest text-zinc-500">Market Scanner</CardTitle>
            </CardHeader>
            <CardContent>
              {signal ? (
                <div className="text-2xl font-bold">{signal.pair}</div>
              ) : (
                <div className="text-2xl font-bold text-zinc-700">SCANNING</div>
              )}
              <p className="text-xs text-zinc-500 mt-1">{signal ? `STRENGTH: ${signal.strength}` : "12 Pairs Active"}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-xl">
            <CardHeader className="pb-2">
              <ShieldCheck className="w-5 h-5 text-emerald-500 mb-2" />
              <CardTitle className="text-sm font-medium uppercase tracking-widest text-zinc-500">Risk Module</CardTitle>
            </CardHeader>
            <CardContent>
              {signal && signal.sl > 0 ? (
                <div className="text-2xl font-bold">{(Math.abs(signal.tp - signal.entry) * 10000).toFixed(0)} Pips</div>
              ) : (
                <div className="text-2xl font-bold">2.0%</div>
              )}
              <p className="text-xs text-zinc-500 mt-1">{signal && signal.sl > 0 ? `SL: ${signal.sl.toFixed(4)}` : "Max Daily Drawdown"}</p>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-xl">
            <CardHeader className="pb-2">
              <Zap className="w-5 h-5 text-emerald-500 mb-2" />
              <CardTitle className="text-sm font-medium uppercase tracking-widest text-zinc-500">Execution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{signal?.latency_ms ? `${signal.latency_ms}ms` : "< 1.0ms"}</div>
              <p className="text-xs text-zinc-500 mt-1">{signal ? "Last Inference Latency" : "Avg Round-trip Latency"}</p>
            </CardContent>
          </Card>
        </div>

        {/* Signal Display Section */}
        {signal && signal.action !== "HOLD" && (
          <div className="mt-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
             <Card className="bg-emerald-500/5 border-emerald-500/20 backdrop-blur-xl overflow-hidden rounded-none border-l-4 border-l-emerald-500">
               <div className="grid grid-cols-1 md:grid-cols-4 items-center">
                 <div className="p-6 border-b md:border-b-0 md:border-r border-emerald-500/10">
                   <div className="flex items-center gap-2 mb-2">
                     <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                     <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-500">Active AI Signal</span>
                   </div>
                   <div className="text-4xl font-black mb-2">{signal.pair}</div>
                   <Badge className={signal.action === 'BUY' ? 'bg-emerald-500 hover:bg-emerald-600 rounded-none' : 'bg-red-500 hover:bg-red-600 rounded-none'}>
                     {signal.action}
                   </Badge>
                 </div>
                 
                 <div className="p-6 border-b md:border-b-0 md:border-r border-emerald-500/10 grid grid-cols-3 gap-4">
                   <div>
                     <div className="text-[10px] text-zinc-500 uppercase font-bold">Entry</div>
                     <div className="font-mono font-bold text-emerald-400">{signal.entry.toFixed(4)}</div>
                   </div>
                   <div>
                     <div className="text-[10px] text-zinc-500 uppercase font-bold">SL</div>
                     <div className="font-mono font-bold text-red-400">{signal.sl.toFixed(4)}</div>
                   </div>
                   <div>
                     <div className="text-[10px] text-zinc-500 uppercase font-bold">TP</div>
                     <div className="font-mono font-bold text-emerald-400">{signal.tp.toFixed(4)}</div>
                   </div>
                 </div>

                 <div className="p-6 border-b md:border-b-0 md:border-r border-emerald-500/10">
                   <div className="text-[10px] text-zinc-500 uppercase font-bold mb-1">Confidence</div>
                   <div className="flex items-end gap-2">
                     <div className="text-3xl font-black text-white">{(signal.confidence * 100).toFixed(1)}%</div>
                   </div>
                   <div className="w-full bg-zinc-800 h-1.5 mt-2 rounded-full overflow-hidden">
                     <div 
                       className="bg-emerald-500 h-full transition-all duration-1000" 
                       style={{ width: `${signal.confidence * 100}%` }}
                     />
                   </div>
                 </div>

                 <div className="p-6">
                   <div className="text-[10px] text-zinc-500 uppercase font-bold mb-1">Reasoning</div>
                   <p className="text-sm text-zinc-300 italic leading-snug">"{signal.reason}"</p>
                 </div>
               </div>
             </Card>
          </div>
        )}

        {/* System Terminal */}
        <div className="mt-12 grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-black border border-zinc-800 rounded-lg overflow-hidden flex flex-col h-[400px]">
              <div className="bg-zinc-900 px-4 py-2 border-b border-zinc-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-emerald-500" />
                  <span className="text-xs font-mono text-zinc-400">trader_console --v 2.0.4</span>
                </div>
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-zinc-800" />
                  <div className="w-2.5 h-2.5 rounded-full bg-zinc-800" />
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                </div>
              </div>
              <div className="flex-1 p-4 font-mono text-xs overflow-y-auto space-y-1">
                {logs.map((log, i) => (
                  <div key={i} className={log?.includes("EXECUTION") ? "text-emerald-400" : "text-zinc-500"}>
                    <span className="text-zinc-700 mr-2">[{new Date().toLocaleTimeString()}]</span>
                    {log}
                  </div>
                ))}
                <div className="animate-pulse inline-block w-2 h-4 bg-emerald-500 ml-1" />
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-500" />
              Top Watchlist
            </h3>
            <div className="space-y-3">
              {(signal?.top_pairs || [
                { pair: "EURUSD", score: 8.4, dir: "UP" },
                { pair: "GBPUSD", score: 7.2, dir: "DOWN" },
                { pair: "XAUUSD", score: 6.8, dir: "UP" },
                { pair: "USDJPY", score: 5.9, dir: "DOWN" },
              ]).map((item, idx) => (
                <div key={idx} className="bg-zinc-900/30 border border-zinc-800 p-4 flex items-center justify-between group hover:border-emerald-500/50 transition-colors text-white">
                  <div>
                    <div className="font-bold">{item.pair}</div>
                    <div className="text-[10px] text-zinc-500 uppercase tracking-tighter">Strength: {item.score}</div>
                  </div>
                  {item.dir === "UP" ? <ArrowUpRight className="text-emerald-500" /> : <ArrowDownRight className="text-red-500" />}
                </div>
              ))}
            </div>

            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-500 mt-1 shrink-0" />
                <p className="text-xs text-red-200/70 leading-relaxed">
                  <strong>DISCLAIMER:</strong> This is a quantitative simulation. Trading forex carries high risk. Past performance does not guarantee future results.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
