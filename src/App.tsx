import React, { useState, useEffect, useCallback } from 'react';
import { Upload, FileImage, Brain, Shield, CheckCircle, XCircle, Loader, Target, Zap, AlertTriangle, RefreshCw } from 'lucide-react';

interface VerificationResult {
  verified: boolean;
  confidence: number;
  message: string;
  max_similarity?: number;
  avg_similarity?: number;
  threshold_used?: number;
  method?: string;
}

interface ModelStatus {
  trained: boolean;
  num_signatures: number;
  model_loaded: boolean;
  verification_threshold?: number;
  performance_metrics?: {
    validation_accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  training_info?: {
    total_training_pairs: number;
    training_time: number;
  };
}

const App: React.FC = () => {
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [referenceFiles, setReferenceFiles] = useState<File[]>([]);
  const [verificationFile, setVerificationFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    trained: false,
    num_signatures: 0,
    model_loaded: false
  });
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<string>('');
  const [connectionStatus, setConnectionStatus] = useState<string>('Checking connection...');
  const [referencesLoaded, setReferencesLoaded] = useState(false);

  // API base URL
  const API_BASE = 'http://localhost:8000';

  // Check model status
  const checkModelStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/model_status`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data: ModelStatus = await response.json();
      console.log('Model status:', data);
      
      setModelStatus(data);
      
      if (data.trained && data.model_loaded) {
        setConnectionStatus('🤖 AI Model Ready');
        if (!trainingProgress || !trainingProgress.includes('ready')) {
          setTrainingProgress('AI model ready for advanced signature verification');
        }
      } else {
        setConnectionStatus('📡 API Connected - Model needs training');
      }
    } catch (error: unknown) {
      console.error('Error checking model status:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setConnectionStatus(`❌ Connection error: ${errorMessage}`);
    }
  }, [API_BASE, trainingProgress]);

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (response.ok) {
        return true;
      }
    } catch (error) {
      return false;
    }
    return false;
  }, [API_BASE]);

  // Initial setup and periodic checks
  useEffect(() => {
    const initializeApp = async () => {
      const healthy = await checkHealth();
      if (healthy) {
        await checkModelStatus();
      }
    };
    
    initializeApp();
    
    const interval = setInterval(async () => {
      if (!isTraining) {
        const healthy = await checkHealth();
        if (healthy) {
          await checkModelStatus();
        }
      }
    }, 8000);
    
    return () => clearInterval(interval);
  }, [checkHealth, checkModelStatus, isTraining]);

  const handleTrainingFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setTrainingFiles(Array.from(files));
    }
  };

  const handleReferenceFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setReferenceFiles(Array.from(files));
    }
  };

  const handleVerificationFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setVerificationFile(file);
    }
  };

  const trainAIModel = async () => {
    if (trainingFiles.length < 5) {
      alert('Please upload at least 5 genuine signature samples for AI training');
      return;
    }

    setIsTraining(true);
    setTrainingProgress('🚀 Initializing AI training pipeline...');
    setVerificationResult(null);
    
    const formData = new FormData();
    trainingFiles.forEach((file) => {
      formData.append('training_images', file);
    });

    try {
      const response = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.trim() && line.startsWith('data: ')) {
              try {
                const jsonData = line.slice(6); // Remove 'data: ' prefix
                const data = JSON.parse(jsonData);
                
                if (data.progress) {
                  setTrainingProgress(`🤖 ${data.progress}`);
                }
                
                if (data.completed || data.status === 'complete' || data.training_finished) {
                  console.log('AI training completion detected:', data);
                  setTrainingProgress(`✅ AI training completed! Model ready for verification.`);
                  setIsTraining(false);
                  // Force model status update
                  setTimeout(() => {
                    checkModelStatus();
                  }, 1000);
                }
                
                if (data.error) {
                  throw new Error(data.error);
                }
              } catch (e: unknown) {
                console.warn('Failed to parse training progress:', e);
              }
            }
          }
        }
      }
    } catch (error: unknown) {
      console.error('Training error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`AI Training failed: ${errorMessage}`);
      setTrainingProgress(`❌ Training failed: ${errorMessage}`);
    } finally {
      setIsTraining(false);
      setTimeout(checkModelStatus, 2000);
    }
  };

  const loadReferenceSignatures = async () => {
    if (referenceFiles.length === 0) {
      alert('Please select reference signatures first');
      return;
    }

    if (!modelStatus.trained) {
      alert('Please train the AI model first');
      return;
    }

    const formData = new FormData();
    referenceFiles.forEach((file) => {
      formData.append('reference_images', file);
    });

    try {
      const response = await fetch(`${API_BASE}/load_reference_signatures`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to load references: ${errorText}`);
      }

      const result = await response.json();
      setReferencesLoaded(true);
      alert(result.message);
    } catch (error: unknown) {
      console.error('Error loading references:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`Failed to load references: ${errorMessage}`);
    }
  };

  const verifySignature = async () => {
    if (!verificationFile) {
      alert('Please upload a signature to verify');
      return;
    }

    if (!modelStatus.trained) {
      alert('Please train the AI model first');
      return;
    }

    setIsVerifying(true);
    setVerificationResult(null);

    const formData = new FormData();
    formData.append('signature_image', verificationFile);

    try {
      const response = await fetch(`${API_BASE}/verify`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || 'Verification failed';
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result: VerificationResult = await response.json();
      setVerificationResult(result);
    } catch (error: unknown) {
      console.error('Verification error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      alert(`AI Verification failed: ${errorMessage}`);
    } finally {
      setIsVerifying(false);
    }
  };

  const resetSystem = async () => {
    if (confirm('Are you sure you want to reset the AI model? This will clear all training data.')) {
      try {
        await fetch(`${API_BASE}/reset_model`, { method: 'DELETE' });
        setTrainingFiles([]);
        setReferenceFiles([]);
        setVerificationFile(null);
        setVerificationResult(null);
        setTrainingProgress('');
        setReferencesLoaded(false);
        await checkModelStatus();
        alert('AI model reset successfully');
      } catch (error) {
        console.error('Reset error:', error);
        alert('Failed to reset model');
      }
    }
  };

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-4 flex items-center justify-center gap-4">
            <div className="relative">
              <Shield className="h-12 w-12 text-cyan-400" />
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-pulse"></div>
            </div>
            AI Signature Verification
          </h1>
          <p className="text-slate-300 text-xl mb-2">Advanced Deep Learning Signature Authentication</p>
          <p className="text-cyan-400 text-sm mb-4">
            Powered by ResNet50 + Siamese Neural Network • Target Accuracy: 90%+
          </p>
          
          {/* Enhanced Connection Status */}
          <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-full text-sm font-medium ${
            connectionStatus.includes('Ready') 
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' 
              : connectionStatus.includes('Connected')
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
              : 'bg-red-500/20 text-red-300 border border-red-500/30'
          }`}>
            <div className={`w-3 h-3 rounded-full animate-pulse ${
              connectionStatus.includes('Ready') ? 'bg-emerald-400' 
              : connectionStatus.includes('Connected') ? 'bg-amber-400' 
              : 'bg-red-400'
            }`} />
            {connectionStatus}
            <button
              onClick={() => {
                checkHealth().then((healthy) => {
                  if (healthy) checkModelStatus();
                });
              }}
              className="ml-2 p-1 rounded-md hover:bg-white/10 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          {/* Performance Metrics */}
          {modelStatus.performance_metrics && (
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                <div className="text-emerald-400 text-lg font-bold">
                  {modelStatus.performance_metrics.validation_accuracy.toFixed(1)}%
                </div>
                <div className="text-slate-400 text-xs">Accuracy</div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                <div className="text-blue-400 text-lg font-bold">
                  {modelStatus.performance_metrics.precision.toFixed(1)}%
                </div>
                <div className="text-slate-400 text-xs">Precision</div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                <div className="text-purple-400 text-lg font-bold">
                  {modelStatus.performance_metrics.recall.toFixed(1)}%
                </div>
                <div className="text-slate-400 text-xs">Recall</div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                <div className="text-cyan-400 text-lg font-bold">
                  {modelStatus.performance_metrics.f1_score.toFixed(1)}%
                </div>
                <div className="text-slate-400 text-xs">F1-Score</div>
              </div>
            </div>
          )}
        </div>

        <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {/* Training Section */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Brain className="h-6 w-6 text-purple-400" />
              <h2 className="text-2xl font-semibold text-white">AI Training</h2>
              <button
                onClick={resetSystem}
                disabled={isTraining}
                className="ml-auto text-sm text-slate-400 hover:text-red-400 disabled:opacity-50 transition-colors"
              >
                Reset
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Upload Genuine Signatures (minimum 5)
                </label>
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:border-purple-400 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleTrainingFileChange}
                    className="hidden"
                    id="training-upload"
                    disabled={isTraining}
                  />
                  <label htmlFor="training-upload" className={`cursor-pointer ${isTraining ? 'cursor-not-allowed opacity-50' : ''}`}>
                    <Upload className="mx-auto h-12 w-12 text-slate-400 mb-2" />
                    <p className="text-slate-300">Upload genuine signatures</p>
                    <p className="text-sm text-slate-500 mt-1">PNG, JPG up to 10MB each</p>
                  </label>
                </div>
              </div>

              {trainingFiles.length > 0 && (
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-sm font-medium text-slate-300 mb-2">
                    Selected files ({trainingFiles.length}):
                  </p>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {trainingFiles.map((file, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm text-slate-400">
                        <FileImage className="h-4 w-4 flex-shrink-0" />
                        <span className="truncate">{file.name}</span>
                        <span className="text-xs">({(file.size / 1024).toFixed(0)}KB)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <button
                onClick={trainAIModel}
                disabled={isTraining || trainingFiles.length < 5}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {isTraining ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    Training AI Model...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4" />
                    Train AI Model
                  </>
                )}
              </button>

              {trainingProgress && (
                <div className={`border rounded-lg p-3 ${
                  trainingProgress.includes('✅') || trainingProgress.includes('ready')
                    ? 'bg-emerald-900/30 border-emerald-500/30'
                    : trainingProgress.includes('❌') || trainingProgress.includes('failed')
                    ? 'bg-red-900/30 border-red-500/30'
                    : 'bg-blue-900/30 border-blue-500/30'
                }`}>
                  <p className={`text-sm ${
                    trainingProgress.includes('✅') || trainingProgress.includes('ready')
                      ? 'text-emerald-300'
                      : trainingProgress.includes('❌') || trainingProgress.includes('failed')
                      ? 'text-red-300'
                      : 'text-blue-300'
                  }`}>
                    {trainingProgress}
                  </p>
                </div>
              )}

              {modelStatus.trained && !isTraining && (
                <div className="bg-emerald-900/30 border border-emerald-500/30 rounded-lg p-3 flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-emerald-400" />
                  <div>
                    <p className="text-sm text-emerald-300 font-medium">AI Model Ready!</p>
                    {modelStatus.training_info && (
                      <p className="text-xs text-emerald-400">
                        {modelStatus.training_info.total_training_pairs} pairs • 
                        {formatTime(modelStatus.training_info.training_time)}
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Reference Loading Section */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Target className="h-6 w-6 text-cyan-400" />
              <h2 className="text-2xl font-semibold text-white">Reference Setup</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Load Reference Signatures
                </label>
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:border-cyan-400 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleReferenceFileChange}
                    className="hidden"
                    id="reference-upload"
                    disabled={!modelStatus.trained}
                  />
                  <label htmlFor="reference-upload" className={`cursor-pointer ${!modelStatus.trained ? 'cursor-not-allowed opacity-50' : ''}`}>
                    <Upload className="mx-auto h-12 w-12 text-slate-400 mb-2" />
                    <p className="text-slate-300">Upload reference signatures</p>
                    <p className="text-sm text-slate-500 mt-1">For comparison during verification</p>
                  </label>
                </div>
              </div>

              {referenceFiles.length > 0 && (
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <p className="text-sm font-medium text-slate-300 mb-2">
                    Reference files ({referenceFiles.length}):
                  </p>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {referenceFiles.map((file, index) => (
                      <div key={index} className="flex items-center gap-2 text-sm text-slate-400">
                        <FileImage className="h-4 w-4 flex-shrink-0" />
                        <span className="truncate">{file.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <button
                onClick={loadReferenceSignatures}
                disabled={!modelStatus.trained || referenceFiles.length === 0}
                className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 text-white py-3 px-4 rounded-lg font-medium hover:from-cyan-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                <Target className="h-4 w-4" />
                Load References
              </button>

              {!modelStatus.trained && (
                <div className="flex items-center gap-2 text-amber-300 bg-amber-900/30 border border-amber-500/30 rounded-lg p-3">
                  <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                  <p className="text-sm">Train the AI model first</p>
                </div>
              )}

              {referencesLoaded && (
                <div className="bg-cyan-900/30 border border-cyan-500/30 rounded-lg p-3 flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-cyan-400" />
                  <p className="text-sm text-cyan-300">References loaded successfully</p>
                </div>
              )}
            </div>
          </div>

          {/* Verification Section */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700 p-6">
            <div className="flex items-center gap-3 mb-6">
              <Shield className="h-6 w-6 text-emerald-400" />
              <h2 className="text-2xl font-semibold text-white">AI Verification</h2>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Upload Signature to Verify
                </label>
                <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:border-emerald-400 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleVerificationFileChange}
                    className="hidden"
                    id="verification-upload"
                    disabled={isVerifying || isTraining}
                  />
                  <label htmlFor="verification-upload" className={`cursor-pointer ${(isVerifying || isTraining) ? 'cursor-not-allowed opacity-50' : ''}`}>
                    <Upload className="mx-auto h-12 w-12 text-slate-400 mb-2" />
                    <p className="text-slate-300">Upload signature for AI verification</p>
                  </label>
                </div>
              </div>

              {verificationFile && (
                <div className="flex items-center gap-2 text-sm text-slate-400 bg-slate-700/50 rounded-lg p-3">
                  <FileImage className="h-4 w-4" />
                  <span className="truncate">{verificationFile.name}</span>
                  <span className="text-xs">({(verificationFile.size / 1024).toFixed(0)}KB)</span>
                </div>
              )}

              <button
                onClick={verifySignature}
                disabled={isVerifying || !verificationFile || !modelStatus.trained || isTraining}
                className="w-full bg-gradient-to-r from-emerald-600 to-green-600 text-white py-3 px-4 rounded-lg font-medium hover:from-emerald-700 hover:to-green-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
              >
                {isVerifying ? (
                  <>
                    <Loader className="h-4 w-4 animate-spin" />
                    AI Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4" />
                    Verify with AI
                  </>
                )}
              </button>

              {/* Verification Results */}
              {verificationResult && (
                <div className={`rounded-lg p-4 border ${
                  verificationResult.verified 
                    ? 'bg-emerald-900/30 border-emerald-500/30' 
                    : 'bg-red-900/30 border-red-500/30'
                }`}>
                  <div className="flex items-center gap-3 mb-4">
                    {verificationResult.verified ? (
                      <CheckCircle className="h-6 w-6 text-emerald-400" />
                    ) : (
                      <XCircle className="h-6 w-6 text-red-400" />
                    )}
                    <span className={`font-bold text-lg ${
                      verificationResult.verified ? 'text-emerald-300' : 'text-red-300'
                    }`}>
                      {verificationResult.verified ? 'VERIFIED ✓' : 'REJECTED ✗'}
                    </span>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-slate-300">AI Confidence:</span>
                      <span className={`font-bold text-lg ${
                        verificationResult.verified ? 'text-emerald-400' : 'text-red-400'
                      }`}>
                        {(verificationResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    
                    <div className="w-full bg-slate-700 rounded-full h-3">
                      <div 
                        className={`h-3 rounded-full transition-all duration-1000 ${
                          verificationResult.verified 
                            ? 'bg-gradient-to-r from-emerald-500 to-green-400' 
                            : 'bg-gradient-to-r from-red-500 to-orange-400'
                        }`}
                        style={{ width: `${verificationResult.confidence * 100}%` }}
                      />
                    </div>
                    
                    {verificationResult.method && (
                      <div className="flex justify-between items-center text-xs text-slate-400">
                        <span>Method:</span>
                        <span className="capitalize">{verificationResult.method.replace('_', ' ')}</span>
                      </div>
                    )}
                    
                    {verificationResult.threshold_used && (
                      <div className="flex justify-between items-center text-xs text-slate-400">
                        <span>Threshold:</span>
                        <span>{(verificationResult.threshold_used * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    
                    <p className={`text-sm mt-3 ${
                      verificationResult.verified ? 'text-emerald-300' : 'text-red-300'
                    }`}>
                      {verificationResult.message}
                    </p>
                  </div>
                </div>
              )}

              {!modelStatus.trained && !isTraining && (
                <div className="flex items-center gap-2 text-amber-300 bg-amber-900/30 border border-amber-500/30 rounded-lg p-3">
                  <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                  <p className="text-sm">AI model needs training first</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-slate-400">
          <p className="text-sm mb-2">
            Advanced AI-Powered Signature Verification System
          </p>
          <p className="text-xs">
            ResNet50 Feature Extraction • Siamese Neural Network • Synthetic Forgery Generation • Advanced Preprocessing
          </p>
          {modelStatus.verification_threshold && (
            <p className="text-xs mt-1 text-slate-500">
              Current AI threshold: {(modelStatus.verification_threshold * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;