import { useState, useCallback } from 'react';
import { toast } from 'sonner';

const API_URL = import.meta.env.VITE_API_URL || '';

export function useTradingData() {
  const [loading, setLoading] = useState(false);
  const [signal, setSignal] = useState(null);
  const [error, setError] = useState(null);

  const fetchSignal = useCallback(async (pair = 'EURUSD') => {
    setLoading(true);
    setError(null);
    try {
      console.log('API URL:', import.meta.env.VITE_API_URL);
      const response = await fetch(`${API_URL}/api/suggestions/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pair }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setSignal(data);
      return data;
    } catch (err) {
      console.log('Fetch Error:', err);
      const msg = err instanceof Error ? err.message : 'Failed to fetch signal';
      setError(msg);
      toast.error(msg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    signal,
    error,
    fetchSignal,
  };
}
