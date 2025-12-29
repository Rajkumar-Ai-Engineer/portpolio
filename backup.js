// Professional AI Portfolio Project Database
// Generated project descriptions for 6 major AI domains

const projectDatabase = {
    // 🧩 Machine Learning Projects
    ml: [
        {
            id: 'ml-1',
            title: 'Sentiment Analysis using LSTM',
            category: 'Machine Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented sentiment analysis using Long Short-Term Memory (LSTM) network to classify text data into positive and negative sentiments. The project explores the power of recurrent neural networks in understanding sequential data with word embeddings for semantic meaning capture.',
            image: 'port/ml/1.jpg',
            video: 'port/ml/1.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'LSTM', 'NLTK', 'Pandas', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '89%',
            modelSize: '12MB',
            trainingTime: '2 hours',
            dataset: 'Custom Text Dataset',
            keyFeatures: [
                'LSTM-based neural network architecture',
                'Word embedding layer for semantic representation',
                'Text preprocessing and tokenization',
                'Hyperparameter tuning optimization',
                'Training progress visualization'
            ],
            technicalDetails: {
                architecture: 'LSTM with Embedding Layer',
                lossFunction: 'Binary Crossentropy',
                optimizer: 'Adam',
                preprocessing: 'Text tokenization and sequence padding',
                evaluation: 'Accuracy and loss curve analysis'
            },
            results: {
                accuracy: '89%',
                precision: '87%',
                recall: '91%',
                f1Score: '89%'
            },
            applications: [
                'Customer feedback analysis',
                'Social media monitoring',
                'Product review classification',
                'Brand sentiment tracking'
            ],
            futurePlans: [
                'Incorporate pre-trained embeddings (GloVe, Word2Vec)',
                'Compare with GRU and Transformer models',
                'Implement BERT for enhanced performance'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/sentiment-analysis-lstm',
            demoLink: 'https://sentiment-lstm-demo.streamlit.app',
            tags: ['LSTM', 'Sentiment Analysis', 'NLP', 'Deep Learning', 'Text Classification'],
            featured: true,
            projectNumber: 1,
            totalProjects: 120,
            categoryProgress: '1/20 ML Projects'
        },
        {
            id: 'ml-2',
            title: 'Customer Segmentation with K-Means Clustering',
            category: 'Machine Learning',
            domain: 'Unsupervised Learning',
            description: 'Utilized K-Means Clustering algorithm to segment customers based on purchasing behavior, identifying distinct groups for personalized marketing strategies. Applied feature scaling and optimal cluster selection using Elbow Method and Silhouette Score.',
            image: 'port/ml/2.jpg',
            video: 'port/ml/2.mp4',
            technologies: ['Python', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn'],
            frameworks: ['Scikit-learn', 'Matplotlib'],
            accuracy: '85%',
            modelSize: '5MB',
            trainingTime: '30 minutes',
            dataset: 'Customer Purchase Dataset',
            keyFeatures: [
                'K-Means clustering algorithm implementation',
                'Feature scaling and preprocessing',
                'Elbow Method for optimal cluster selection',
                'Silhouette Score evaluation',
                'Customer behavior pattern analysis'
            ],
            technicalDetails: {
                architecture: 'K-Means Clustering',
                evaluation: 'Elbow Method + Silhouette Score',
                preprocessing: 'Feature scaling and missing value handling',
                optimization: 'Optimal cluster number determination',
                visualization: 'Cluster visualization and analysis'
            },
            results: {
                clusterAccuracy: '85%',
                silhouetteScore: '0.72',
                optimalClusters: '5 segments',
                customerInsights: 'Distinct purchasing patterns identified'
            },
            applications: [
                'Personalized marketing campaigns',
                'Customer retention strategies',
                'Product recommendation systems',
                'Targeted advertising'
            ],
            futurePlans: [
                'Experiment with DBSCAN and Hierarchical Clustering',
                'Implement t-SNE and PCA for visualization',
                'Combine with RFM analysis for enhanced segmentation'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/customer-segmentation-kmeans',
            demoLink: 'https://customer-segmentation-demo.streamlit.app',
            tags: ['K-Means', 'Customer Segmentation', 'Unsupervised Learning', 'Clustering', 'Marketing Analytics'],
            featured: true,
            projectNumber: 2,
            totalProjects: 120,
            categoryProgress: '2/20 ML Projects'
        },
        {
            id: 'ml-3',
            title: 'Stock Price Prediction with Linear Regression',
            category: 'Machine Learning',
            domain: 'Financial Forecasting',
            description: 'Implemented Linear Regression model to predict Tesla stock prices using historical data including opening price, closing price, volume, and adjusted close prices. Applied time-series analysis for quantitative finance applications.',
            image: 'port/ml/3.jpg',
            video: 'port/ml/3.mp4',
            technologies: ['Python', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'yfinance'],
            frameworks: ['Scikit-learn', 'Matplotlib'],
            accuracy: '78%',
            modelSize: '2MB',
            trainingTime: '15 minutes',
            dataset: 'Tesla Stock Historical Data',
            keyFeatures: [
                'Linear regression model implementation',
                'Historical stock data analysis',
                'Time-series data preprocessing',
                'Feature engineering with stock indicators',
                'Performance evaluation with MAE and RMSE'
            ],
            technicalDetails: {
                architecture: 'Linear Regression',
                evaluation: 'MAE and RMSE metrics',
                preprocessing: 'Data scaling and feature selection',
                features: 'Open, Close, Volume, Adjusted Close',
                timeframe: 'Historical stock data analysis'
            },
            results: {
                accuracy: '78%',
                mae: '12.5',
                rmse: '18.3',
                r2Score: '0.76'
            },
            applications: [
                'Algorithmic trading strategies',
                'Investment decision support',
                'Portfolio optimization',
                'Risk assessment'
            ],
            futurePlans: [
                'Implement ARIMA and LSTM models',
                'Integrate technical indicators (RSI, Moving Averages)',
                'Explore market sentiment analysis integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/stock-price-prediction-lr',
            demoLink: 'https://tesla-stock-prediction-demo.streamlit.app',
            tags: ['Linear Regression', 'Stock Prediction', 'Financial Forecasting', 'Tesla', 'Time Series'],
            featured: true,
            projectNumber: 3,
            totalProjects: 120,
            categoryProgress: '3/20 ML Projects'
        },
        {
            id: 'ml-4',
            title: 'Image Classification – Dog vs. Cat',
            category: 'Machine Learning',
            domain: 'Computer Vision',
            description: 'Applied Convolutional Neural Network (CNN) for binary image classification to distinguish between dogs and cats. Utilized the popular Kaggle dataset with data augmentation techniques for improved model generalization and performance.',
            image: 'port/ml/4.jpg',
            video: 'port/ml/4.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'CNN', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '92%',
            modelSize: '25MB',
            trainingTime: '3 hours',
            dataset: 'Kaggle Dogs vs Cats Dataset',
            keyFeatures: [
                'Convolutional Neural Network architecture',
                'Data augmentation for generalization',
                'Feature extraction and classification',
                'Training curve visualization',
                'Hyperparameter tuning optimization'
            ],
            technicalDetails: {
                architecture: 'Custom CNN with Conv2D layers',
                lossFunction: 'Binary Crossentropy',
                optimizer: 'Adam',
                preprocessing: 'Image normalization and augmentation',
                evaluation: 'Accuracy and loss curve analysis'
            },
            results: {
                accuracy: '92%',
                precision: '91%',
                recall: '93%',
                f1Score: '92%'
            },
            applications: [
                'Pet recognition systems',
                'Animal classification apps',
                'Veterinary diagnostic tools',
                'Wildlife monitoring'
            ],
            futurePlans: [
                'Implement ResNet and VGG architectures',
                'Apply transfer learning with pre-trained models',
                'Expand to multi-class animal classification'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/dog-cat-classification-cnn',
            demoLink: 'https://dog-cat-classifier-demo.streamlit.app',
            tags: ['CNN', 'Image Classification', 'Computer Vision', 'Dogs vs Cats', 'Deep Learning'],
            featured: true,
            projectNumber: 4,
            totalProjects: 120,
            categoryProgress: '4/20 ML Projects'
        },
        {
            id: 'ml-5',
            title: 'Hand Gesture Recognition using MediaPipe',
            category: 'Machine Learning',
            domain: 'Computer Vision',
            description: 'Implemented real-time hand gesture recognition using MediaPipe for hand tracking and K-Nearest Neighbors classifier for gesture classification. Enabled natural device interaction through webcam-based gesture control with immediate feedback.',
            image: 'port/ml/5.jpg',
            video: 'port/ml/5.mp4',
            technologies: ['Python', 'MediaPipe', 'OpenCV', 'KNN', 'NumPy', 'Scikit-learn'],
            frameworks: ['MediaPipe', 'OpenCV', 'Scikit-learn'],
            accuracy: '94%',
            modelSize: '8MB',
            trainingTime: '45 minutes',
            dataset: 'Custom Hand Gesture Dataset',
            keyFeatures: [
                'MediaPipe Hand Landmarks Model integration',
                'Real-time webcam gesture recognition',
                'K-Nearest Neighbors classification',
                'Hand landmark preprocessing and normalization',
                'Immediate gesture feedback system'
            ],
            technicalDetails: {
                architecture: 'MediaPipe + KNN Classifier',
                preprocessing: 'Hand landmark extraction and normalization',
                realTimeProcessing: 'Webcam input with live recognition',
                classification: 'Spatial data classification using KNN',
                evaluation: 'Real-time accuracy assessment'
            },
            results: {
                accuracy: '94%',
                realTimePerformance: '30 FPS',
                gestureLatency: '50ms',
                supportedGestures: '8 different gestures'
            },
            applications: [
                'Human-computer interaction',
                'Virtual reality control',
                'Assistive technology',
                'Hands-free device control'
            ],
            futurePlans: [
                'Implement CNN and LSTM architectures',
                'Develop sign language translation system',
                'Integrate with IoT devices for smart home control'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/hand-gesture-recognition-mediapipe',
            demoLink: 'https://hand-gesture-demo.streamlit.app',
            tags: ['MediaPipe', 'Hand Gesture Recognition', 'Computer Vision', 'KNN', 'Real-time Processing'],
            featured: true,
            projectNumber: 5,
            totalProjects: 120,
            categoryProgress: '5/20 ML Projects'
        },
        {
            id: 'ml-6',
            title: 'Predictive Analysis for Healthcare – Lung Cancer Detection',
            category: 'Machine Learning',
            domain: 'Healthcare Analytics',
            description: 'Implemented Random Forest Classifier for early lung cancer detection using healthcare datasets. Applied extensive data preprocessing, hyperparameter tuning, and feature importance analysis to achieve reliable predictions for medical diagnosis support.',
            image: 'port/ml/6.jpg',
            video: 'port/ml/6.mp4',
            technologies: ['Python', 'Random Forest', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn'],
            frameworks: ['Scikit-learn', 'Matplotlib'],
            accuracy: '96%',
            modelSize: '15MB',
            trainingTime: '2 hours',
            dataset: 'Healthcare Lung Cancer Dataset',
            keyFeatures: [
                'Random Forest Classifier implementation',
                'Extensive data preprocessing pipeline',
                'Hyperparameter tuning with Grid Search',
                'Cross-validation for model reliability',
                'Feature importance analysis'
            ],
            technicalDetails: {
                architecture: 'Random Forest Classifier',
                preprocessing: 'Missing value handling, outlier detection, feature scaling',
                optimization: 'Grid Search with Cross-Validation',
                evaluation: 'Accuracy, Precision, Recall, F1-Score',
                featureAnalysis: 'Feature importance ranking'
            },
            results: {
                accuracy: '96%',
                precision: '94%',
                recall: '97%',
                f1Score: '95%'
            },
            applications: [
                'Early cancer detection systems',
                'Medical diagnosis support tools',
                'Healthcare risk assessment',
                'Patient screening automation'
            ],
            futurePlans: [
                'Implement Gradient Boosting and XGBoost',
                'Extend to survival analysis predictions',
                'Deploy healthcare dashboard for professionals'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/lung-cancer-detection-rf',
            demoLink: 'https://lung-cancer-prediction-demo.streamlit.app',
            tags: ['Random Forest', 'Healthcare Analytics', 'Cancer Detection', 'Medical AI', 'Predictive Analysis'],
            featured: true,
            projectNumber: 6,
            totalProjects: 120,
            categoryProgress: '6/20 ML Projects'
        },
        {
            id: 'ml-7',
            title: 'Time Series Forecasting using ARIMA and TSA Components',
            category: 'Machine Learning',
            domain: 'Time Series Analysis',
            description: 'Implemented ARIMA model for time series forecasting with comprehensive TSA component analysis including trend, seasonality, and residuals. Applied ACF and PACF analysis for optimal parameter tuning and accurate future value predictions.',
            image: 'port/ml/7.jpg',
            video: 'port/ml/7.mp4',
            technologies: ['Python', 'ARIMA', 'Statsmodels', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn'],
            frameworks: ['Statsmodels', 'Matplotlib'],
            accuracy: '87%',
            modelSize: '5MB',
            trainingTime: '1 hour',
            dataset: 'Time Series Sales/Stock Dataset',
            keyFeatures: [
                'ARIMA model implementation with parameter tuning',
                'Time series decomposition (trend, seasonality, residuals)',
                'ACF and PACF analysis for lag determination',
                'Stationarity testing and data transformation',
                'Out-of-sample forecasting and validation'
            ],
            technicalDetails: {
                architecture: 'ARIMA (p, d, q) Model',
                preprocessing: 'Stationarity transformation and differencing',
                parameterTuning: 'ACF/PACF analysis for optimal (p,d,q)',
                evaluation: 'MAE, RMSE, and forecast accuracy',
                validation: 'Time series cross-validation'
            },
            results: {
                accuracy: '87%',
                mae: '8.2',
                rmse: '12.5',
                forecastHorizon: '12 periods ahead'
            },
            applications: [
                'Sales demand forecasting',
                'Stock price prediction',
                'Weather pattern analysis',
                'Economic indicator forecasting'
            ],
            futurePlans: [
                'Implement SARIMA for seasonal patterns',
                'Explore Prophet and LSTM models',
                'Develop real-time forecasting dashboard'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/time-series-arima-forecasting',
            demoLink: 'https://arima-forecasting-demo.streamlit.app',
            tags: ['ARIMA', 'Time Series Forecasting', 'TSA', 'Statistical Modeling', 'Predictive Analytics'],
            featured: true,
            projectNumber: 7,
            totalProjects: 120,
            categoryProgress: '7/20 ML Projects'
        },
        {
            id: 'ml-8',
            title: 'Credit Card Fraud Detection using Logistic Regression',
            category: 'Machine Learning',
            domain: 'Financial Security',
            description: 'Implemented Logistic Regression for credit card fraud detection with SMOTE for handling imbalanced datasets. Deployed as a real-time web application using FastAPI and Uvicorn for instant fraud prediction and transaction monitoring.',
            image: 'port/ml/8.jpg',
            video: 'port/ml/8.mp4',
            technologies: ['Python', 'Logistic Regression', 'Scikit-learn', 'SMOTE', 'FastAPI', 'Uvicorn', 'Pickle'],
            frameworks: ['Scikit-learn', 'FastAPI', 'Uvicorn'],
            accuracy: '94%',
            modelSize: '3MB',
            trainingTime: '45 minutes',
            dataset: 'Credit Card Fraud Dataset (Imbalanced)',
            keyFeatures: [
                'Logistic Regression classification model',
                'SMOTE for imbalanced dataset handling',
                'Model serialization with Pickle',
                'FastAPI web application deployment',
                'Real-time fraud prediction system'
            ],
            technicalDetails: {
                architecture: 'Logistic Regression Classifier',
                preprocessing: 'SMOTE oversampling and feature scaling',
                evaluation: 'Precision, Recall, F1-Score, ROC-AUC',
                deployment: 'FastAPI + Uvicorn web application',
                serialization: 'Pickle model persistence'
            },
            results: {
                accuracy: '94%',
                precision: '92%',
                recall: '96%',
                f1Score: '94%',
                rocAuc: '0.97'
            },
            applications: [
                'Real-time fraud detection systems',
                'Financial transaction monitoring',
                'Banking security applications',
                'Payment gateway protection'
            ],
            futurePlans: [
                'Upgrade UI with Streamlit or Dash',
                'Implement Gradient Boosting and Neural Networks',
                'Scale for live transaction system integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/credit-card-fraud-detection',
            demoLink: 'https://fraud-detection-fastapi-demo.herokuapp.com',
            tags: ['Logistic Regression', 'Fraud Detection', 'SMOTE', 'FastAPI', 'Financial Security'],
            featured: true,
            projectNumber: 8,
            totalProjects: 120,
            categoryProgress: '8/20 ML Projects'
        },
        {
            id: 'ml-9',
            title: 'Movie Recommendation System using SVD',
            category: 'Machine Learning',
            domain: 'Recommendation Systems',
            description: 'Built movie recommendation system using Singular Value Decomposition (SVD) for collaborative filtering. Applied matrix factorization to decompose user-item rating matrix into latent factors for personalized movie recommendations.',
            image: 'port/ml/9.jpg',
            video: 'port/ml/9.mp4',
            technologies: ['Python', 'SVD', 'NumPy', 'Pandas', 'Scikit-learn', 'Surprise', 'Matplotlib'],
            frameworks: ['Surprise', 'Scikit-learn'],
            accuracy: '91%',
            modelSize: '20MB',
            trainingTime: '1.5 hours',
            dataset: 'MovieLens Dataset',
            keyFeatures: [
                'Singular Value Decomposition implementation',
                'User-item rating matrix construction',
                'Collaborative filtering algorithm',
                'Missing rating prediction',
                'Top-N movie recommendation generation'
            ],
            technicalDetails: {
                architecture: 'SVD Matrix Factorization',
                preprocessing: 'User-item matrix creation and normalization',
                evaluation: 'RMSE, MAE, and recommendation accuracy',
                coldStart: 'Cold start problem analysis',
                latentFactors: 'User and item latent feature extraction'
            },
            results: {
                accuracy: '91%',
                rmse: '0.87',
                mae: '0.68',
                recommendationPrecision: '89%'
            },
            applications: [
                'Movie streaming platforms',
                'Content discovery systems',
                'Personalized entertainment apps',
                'E-commerce product recommendations'
            ],
            futurePlans: [
                'Implement hybrid recommendation models',
                'Explore Neural Collaborative Filtering',
                'Deploy as interactive web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/movie-recommendation-svd',
            demoLink: 'https://movie-recommender-svd-demo.streamlit.app',
            tags: ['SVD', 'Recommendation System', 'Collaborative Filtering', 'Matrix Factorization', 'MovieLens'],
            featured: true,
            projectNumber: 9,
            totalProjects: 120,
            categoryProgress: '9/20 ML Projects'
        },
        {
            id: 'ml-10',
            title: 'Customer Churn Prediction',
            category: 'Machine Learning',
            domain: 'Business Analytics',
            description: 'Developed customer churn prediction model to identify customers likely to leave service subscriptions. Applied multiple classification algorithms with SMOTE for imbalanced data handling and comprehensive feature engineering for business insights.',
            image: 'port/ml/10.jpg',
            video: 'port/ml/10.mp4',
            technologies: ['Python', 'XGBoost', 'Random Forest', 'Logistic Regression', 'SMOTE', 'Scikit-learn', 'Pandas'],
            frameworks: ['Scikit-learn', 'XGBoost'],
            accuracy: '93%',
            modelSize: '18MB',
            trainingTime: '2 hours',
            dataset: 'Customer Behavior Dataset',
            keyFeatures: [
                'Multiple classification algorithm comparison',
                'SMOTE for imbalanced dataset handling',
                'Grid Search and Cross-Validation optimization',
                'Comprehensive feature engineering',
                'Business insight visualization'
            ],
            technicalDetails: {
                architecture: 'Ensemble Methods (XGBoost, Random Forest)',
                preprocessing: 'SMOTE oversampling and feature scaling',
                optimization: 'Grid Search with Cross-Validation',
                evaluation: 'Precision, Recall, F1-Score, ROC-AUC',
                featureEngineering: 'Demographics and usage pattern analysis'
            },
            results: {
                accuracy: '93%',
                precision: '91%',
                recall: '95%',
                f1Score: '93%',
                rocAuc: '0.96'
            },
            applications: [
                'Telecommunications customer retention',
                'SaaS subscription management',
                'Retail customer loyalty programs',
                'Banking customer relationship management'
            ],
            futurePlans: [
                'Integrate real-time monitoring dashboard',
                'Explore Deep Learning and CLV modeling',
                'Collaborate with domain experts for strategy optimization'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/customer-churn-prediction',
            demoLink: 'https://churn-prediction-demo.streamlit.app',
            tags: ['Churn Prediction', 'XGBoost', 'Business Analytics', 'Customer Retention', 'SMOTE'],
            featured: true,
            projectNumber: 10,
            totalProjects: 120,
            categoryProgress: '10/20 ML Projects'
        },
        {
            id: 'ml-11',
            title: 'Face Recognition System',
            category: 'Machine Learning',
            domain: 'Computer Vision',
            description: 'Implemented face recognition system using Face Recognition library and OpenCV for real-time biometric identification. Applied facial embeddings with Euclidean distance matching for high-accuracy face verification and authentication.',
            image: 'port/ml/11.jpg',
            video: 'port/ml/11.mp4',
            technologies: ['Python', 'Face Recognition Library', 'OpenCV', 'dlib', 'NumPy', 'Pickle'],
            frameworks: ['Face Recognition', 'OpenCV', 'dlib'],
            accuracy: '96%',
            modelSize: '45MB',
            trainingTime: '1 hour',
            dataset: 'Custom Face Database',
            keyFeatures: [
                'Pre-trained deep learning facial embeddings',
                'Real-time video processing with OpenCV',
                'Euclidean distance-based face matching',
                'Multi-face detection and recognition',
                'Live webcam integration'
            ],
            technicalDetails: {
                architecture: 'Face Recognition Library + OpenCV',
                embeddings: 'Pre-trained deep learning facial features',
                matching: 'Euclidean distance similarity measurement',
                realTimeProcessing: 'Live video stream analysis',
                storage: 'Face database with pickle serialization'
            },
            results: {
                accuracy: '96%',
                recognitionSpeed: '15 FPS',
                falsePositiveRate: '2%',
                multiplefaces: 'Up to 10 faces simultaneously'
            },
            applications: [
                'Security and access control systems',
                'Attendance tracking automation',
                'User authentication systems',
                'Smart surveillance solutions'
            ],
            futurePlans: [
                'Implement clustering for large datasets',
                'Explore PyTorch/TensorFlow deep learning approaches',
                'Deploy on Raspberry Pi for edge computing'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/face-recognition-system',
            demoLink: 'https://face-recognition-demo.streamlit.app',
            tags: ['Face Recognition', 'Computer Vision', 'Biometric Security', 'OpenCV', 'Real-time Processing'],
            featured: true,
            projectNumber: 11,
            totalProjects: 120,
            categoryProgress: '11/20 ML Projects'
        },
        {
            id: 'ml-12',
            title: 'Multi-Label Image Classification with CNN',
            category: 'Machine Learning',
            domain: 'Computer Vision',
            description: 'Developed multi-label image classification system for food recognition using custom CNN architecture. Applied sigmoid activation for multiple label prediction including cuisine type, preparation method, and dietary classifications.',
            image: 'port/ml/12.jpg',
            video: 'port/ml/12.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'CNN', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '88%',
            modelSize: '35MB',
            trainingTime: '4 hours',
            dataset: 'Multi-Label Food Dataset',
            keyFeatures: [
                'Custom CNN architecture for multi-label classification',
                'Sigmoid activation for multiple label prediction',
                'Data augmentation for class imbalance handling',
                'Binary cross-entropy loss optimization',
                'Batch normalization and dropout regularization'
            ],
            technicalDetails: {
                architecture: 'Custom CNN with Sigmoid Output Layer',
                lossFunction: 'Binary Cross-Entropy',
                optimizer: 'Adam',
                preprocessing: 'Data augmentation and label encoding',
                evaluation: 'Precision, Recall, F1-Score, mAP'
            },
            results: {
                accuracy: '88%',
                meanAveragePrecision: '0.85',
                precision: '86%',
                recall: '90%',
                f1Score: '88%'
            },
            applications: [
                'Food recognition and nutrition apps',
                'Restaurant menu digitization',
                'Dietary tracking systems',
                'E-commerce product categorization'
            ],
            futurePlans: [
                'Implement transfer learning with ResNet/EfficientNet',
                'Deploy as real-time web application',
                'Explore semi-supervised learning approaches'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/multi-label-food-classification',
            demoLink: 'https://food-classifier-demo.streamlit.app',
            tags: ['Multi-Label Classification', 'CNN', 'Food Recognition', 'Computer Vision', 'Deep Learning'],
            featured: true,
            projectNumber: 12,
            totalProjects: 120,
            categoryProgress: '12/20 ML Projects'
        },
        {
            id: 'ml-13',
            title: 'Traffic Flow Prediction with DNN',
            category: 'Machine Learning',
            domain: 'Smart Cities',
            description: 'Developed traffic flow prediction system using Deep Neural Network for forecasting traffic patterns based on historical data. Implemented advanced callbacks including EarlyStopping, ModelCheckpoint, and TensorBoard for optimized training.',
            image: 'port/ml/13.jpg',
            video: 'port/ml/13.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'DNN', 'TensorBoard', 'NumPy', 'Pandas'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '85%',
            modelSize: '22MB',
            trainingTime: '3 hours',
            dataset: 'Historical Traffic Flow Dataset',
            keyFeatures: [
                'Custom Deep Neural Network architecture',
                'EarlyStopping callback for overfitting prevention',
                'ModelCheckpoint for best model preservation',
                'TensorBoard visualization for training monitoring',
                'Time-series data preprocessing and normalization'
            ],
            technicalDetails: {
                architecture: 'Multi-layer Dense Neural Network',
                activationFunctions: 'ReLU (hidden), Linear (output)',
                callbacks: 'EarlyStopping, ModelCheckpoint, TensorBoard',
                preprocessing: 'Data normalization and feature scaling',
                evaluation: 'MAE, RMSE for regression performance'
            },
            results: {
                accuracy: '85%',
                mae: '12.3',
                rmse: '18.7',
                trainingEfficiency: 'Optimized with callbacks'
            },
            applications: [
                'Smart city traffic management',
                'Urban transportation planning',
                'Congestion prediction systems',
                'Logistics route optimization'
            ],
            futurePlans: [
                'Integrate real-time traffic APIs',
                'Include external factors (incidents, events)',
                'Explore RNN/LSTM for sequential data'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/traffic-flow-prediction-dnn',
            demoLink: 'https://traffic-prediction-demo.streamlit.app',
            tags: ['DNN', 'Traffic Prediction', 'Smart Cities', 'TensorBoard', 'Time Series'],
            featured: true,
            projectNumber: 13,
            totalProjects: 120,
            categoryProgress: '13/20 ML Projects'
        },
        {
            id: 'ml-14',
            title: 'Text Classification with BERT and Fine-Tuning',
            category: 'Machine Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented text classification using BERT fine-tuning on AG News Dataset for categorizing news articles into World, Sports, Business, and Sci/Tech. Applied Hugging Face Transformers with custom TrainingArguments for optimal performance.',
            image: 'port/ml/14.jpg',
            video: 'port/ml/14.mp4',
            technologies: ['Python', 'BERT', 'Transformers', 'Hugging Face', 'PyTorch', 'Pandas', 'NumPy'],
            frameworks: ['Hugging Face Transformers', 'PyTorch'],
            accuracy: '94%',
            modelSize: '440MB',
            trainingTime: '5 hours',
            dataset: 'AG News Dataset',
            keyFeatures: [
                'BERT pre-trained model fine-tuning',
                'Hugging Face Transformers integration',
                'Custom TrainingArguments configuration',
                'Multi-class news categorization',
                'Advanced tokenization and preprocessing'
            ],
            technicalDetails: {
                architecture: 'BERT with Classification Head',
                finetuning: 'Transfer learning with custom classification layer',
                training: 'Trainer API with TrainingArguments',
                preprocessing: 'BERT tokenization and input formatting',
                evaluation: 'Accuracy, F1-Score, Confusion Matrix'
            },
            results: {
                accuracy: '94%',
                f1Score: '93%',
                precision: '94%',
                recall: '93%'
            },
            applications: [
                'News categorization systems',
                'Content filtering platforms',
                'Document classification',
                'Automated content tagging'
            ],
            futurePlans: [
                'Apply to multi-label classification tasks',
                'Experiment with RoBERTa and DistilBERT',
                'Deploy real-time news categorization system'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/bert-text-classification',
            demoLink: 'https://bert-news-classifier-demo.streamlit.app',
            tags: ['BERT', 'Text Classification', 'Fine-tuning', 'Transformers', 'NLP'],
            featured: true,
            projectNumber: 14,
            totalProjects: 120,
            categoryProgress: '14/20 ML Projects'
        },
        {
            id: 'ml-15',
            title: 'Handwritten Digit Classification with CNN',
            category: 'Machine Learning',
            domain: 'Computer Vision',
            description: 'Implemented Convolutional Neural Network for handwritten digit classification using MNIST dataset. Applied convolutional layers for feature extraction, pooling for dimensionality reduction, and achieved high accuracy in digit recognition.',
            image: 'port/ml/15.jpg',
            video: 'port/ml/15.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'CNN', 'MNIST', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '98%',
            modelSize: '12MB',
            trainingTime: '2 hours',
            dataset: 'MNIST Handwritten Digits',
            keyFeatures: [
                'Convolutional Neural Network architecture',
                'Spatial feature extraction with Conv2D layers',
                'MaxPooling for dimensionality reduction',
                'ReLU activation and Softmax classification',
                'Data normalization and preprocessing'
            ],
            technicalDetails: {
                architecture: 'CNN with Conv2D, MaxPooling, Dense layers',
                lossFunction: 'Categorical Cross-Entropy',
                optimizer: 'Adam',
                preprocessing: 'Pixel normalization (0-1 scaling)',
                evaluation: 'Test accuracy and prediction visualization'
            },
            results: {
                accuracy: '98%',
                precision: '98%',
                recall: '98%',
                f1Score: '98%'
            },
            applications: [
                'Postal code recognition systems',
                'Bank check processing',
                'Educational assessment tools',
                'Digital form automation'
            ],
            futurePlans: [
                'Extend to multi-digit recognition (EMNIST)',
                'Experiment with ResNet and MobileNet',
                'Deploy as real-time digit recognition app'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/mnist-digit-classification-cnn',
            demoLink: 'https://digit-classifier-demo.streamlit.app',
            tags: ['CNN', 'MNIST', 'Digit Classification', 'Computer Vision', 'Deep Learning'],
            featured: true,
            projectNumber: 15,
            totalProjects: 120,
            categoryProgress: '15/20 ML Projects'
        },
        {
            id: 'ml-16',
            title: 'Sales Forecasting with XGBoost',
            category: 'Machine Learning',
            domain: 'Business Analytics',
            description: 'Built sales forecasting model using XGBoost for predicting future sales trends and optimizing inventory management. Applied hyperparameter tuning and feature importance analysis for business-driven insights.',
            image: 'port/ml/16.jpg',
            video: 'port/ml/16.mp4',
            technologies: ['Python', 'XGBoost', 'Pandas', 'NumPy', 'Scikit-learn', 'Matplotlib', 'Seaborn'],
            frameworks: ['XGBoost', 'Scikit-learn'],
            accuracy: '91%',
            modelSize: '25MB',
            trainingTime: '1.5 hours',
            dataset: 'Historical Sales Dataset',
            keyFeatures: [
                'XGBoost gradient boosting implementation',
                'Hyperparameter tuning optimization',
                'Cross-validation for model generalization',
                'Feature importance analysis',
                'Seasonal trend prediction capability'
            ],
            technicalDetails: {
                architecture: 'XGBoost Gradient Boosting',
                preprocessing: 'Missing value handling, categorical encoding, scaling',
                optimization: 'Learning rate, max depth, n_estimators tuning',
                evaluation: 'RMSE, R² Score, Cross-validation',
                featureAnalysis: 'Business driver identification'
            },
            results: {
                accuracy: '91%',
                rmse: '15.2',
                r2Score: '0.89',
                crossValidationScore: '88%'
            },
            applications: [
                'Inventory management optimization',
                'Revenue forecasting systems',
                'Promotional planning tools',
                'Supply chain management'
            ],
            futurePlans: [
                'Include external factors (weather, economic indicators)',
                'Deploy real-time forecasting system',
                'Compare with LSTM forecasting models'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/sales-forecasting-xgboost',
            demoLink: 'https://sales-forecast-demo.streamlit.app',
            tags: ['XGBoost', 'Sales Forecasting', 'Business Analytics', 'Time Series', 'Feature Importance'],
            featured: true,
            projectNumber: 16,
            totalProjects: 120,
            categoryProgress: '16/20 ML Projects'
        },
        {
            id: 'ml-17',
            title: 'Plant Disease Classification with CNN',
            category: 'Machine Learning',
            domain: 'Agricultural AI',
            description: 'Developed plant disease classification model using CNN for early disease detection in crops. Applied data augmentation and preprocessing on Plant Village Dataset to assist farmers in disease diagnosis and crop yield optimization.',
            image: 'port/ml/17.jpg',
            video: 'port/ml/17.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'CNN', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '95%',
            modelSize: '40MB',
            trainingTime: '4 hours',
            dataset: 'Plant Village Dataset',
            keyFeatures: [
                'CNN architecture for plant disease classification',
                'Data augmentation for improved generalization',
                'Multi-class disease category prediction',
                'Image preprocessing and normalization',
                'Real-time agricultural disease detection'
            ],
            technicalDetails: {
                architecture: 'CNN with Conv2D, MaxPooling, Dense layers',
                lossFunction: 'Categorical Cross-Entropy',
                optimizer: 'Adam',
                preprocessing: 'Image resizing, normalization, augmentation',
                evaluation: 'Accuracy, Confusion Matrix, Prediction visualization'
            },
            results: {
                accuracy: '95%',
                precision: '94%',
                recall: '96%',
                f1Score: '95%'
            },
            applications: [
                'Agricultural disease management systems',
                'Mobile crop monitoring apps',
                'IoT-based field monitoring',
                'Precision agriculture solutions'
            ],
            futurePlans: [
                'Integrate with mobile app for field use',
                'Expand dataset to include more crops',
                'Compare with EfficientNet and ResNet architectures'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/plant-disease-classification-cnn',
            demoLink: 'https://plant-disease-classifier-demo.streamlit.app',
            tags: ['CNN', 'Plant Disease Classification', 'Agricultural AI', 'Computer Vision', 'Crop Management'],
            featured: true,
            projectNumber: 17,
            totalProjects: 120,
            categoryProgress: '17/20 ML Projects'
        },
        {
            id: 'ml-18',
            title: 'Loan Approval Prediction with Random Forest',
            category: 'Machine Learning',
            domain: 'Financial Services',
            description: 'Built loan approval prediction model using Random Forest algorithm to assess loan eligibility based on applicant information. Applied ensemble learning with hyperparameter tuning and feature importance analysis for financial decision support.',
            image: 'port/ml/18.jpg',
            video: 'port/ml/18.mp4',
            technologies: ['Python', 'Random Forest', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn'],
            frameworks: ['Scikit-learn'],
            accuracy: '89%',
            modelSize: '15MB',
            trainingTime: '45 minutes',
            dataset: 'Loan Approval Dataset',
            keyFeatures: [
                'Random Forest ensemble learning implementation',
                'Mixed data type preprocessing (categorical/numerical)',
                'Hyperparameter tuning optimization',
                'Cross-validation for model stability',
                'Feature importance analysis for business insights'
            ],
            technicalDetails: {
                architecture: 'Random Forest Classifier',
                preprocessing: 'Missing value handling, categorical encoding, feature scaling',
                optimization: 'Number of trees, max depth, min samples tuning',
                evaluation: 'Accuracy, Precision, Recall, F1-Score',
                validation: 'Cross-validation for generalization assessment'
            },
            results: {
                accuracy: '89%',
                precision: '87%',
                recall: '91%',
                f1Score: '89%'
            },
            applications: [
                'Automated loan processing systems',
                'Credit risk assessment tools',
                'Financial decision support systems',
                'Banking application screening'
            ],
            futurePlans: [
                'Include external economic factors',
                'Develop user-friendly prediction interface',
                'Compare with XGBoost and LightGBM models'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/loan-approval-prediction-rf',
            demoLink: 'https://loan-approval-demo.streamlit.app',
            tags: ['Random Forest', 'Loan Approval', 'Financial Services', 'Ensemble Learning', 'Credit Assessment'],
            featured: true,
            projectNumber: 18,
            totalProjects: 120,
            categoryProgress: '18/20 ML Projects'
        },
        {
            id: 'ml-19',
            title: 'Predictive Maintenance for Milling Machines',
            category: 'Machine Learning',
            domain: 'Industrial IoT',
            description: 'Built predictive maintenance model using MLPClassifier to identify potential milling machine failures before they occur. Applied SMOTE for imbalanced data handling and multiple encoding techniques for optimal machinery performance prediction.',
            image: 'port/ml/19.jpg',
            video: 'port/ml/19.mp4',
            technologies: ['Python', 'MLPClassifier', 'SMOTE', 'LabelEncoder', 'FrequencyEncoder', 'TargetEncoder', 'StandardScaler'],
            frameworks: ['Scikit-learn', 'Imbalanced-learn'],
            accuracy: '92%',
            modelSize: '28MB',
            trainingTime: '2.5 hours',
            dataset: 'Milling Machine Performance Dataset',
            keyFeatures: [
                'MLPClassifier deep learning implementation',
                'SMOTE for imbalanced data oversampling',
                'Multiple encoding techniques (Label, Frequency, Target)',
                'StandardScaler for feature normalization',
                'Custom estimator HTML documentation'
            ],
            technicalDetails: {
                architecture: 'Multi-Layer Perceptron Classifier',
                preprocessing: 'SMOTE oversampling, multiple encoders, scaling',
                optimization: 'Hidden layers, activation functions, learning rate tuning',
                evaluation: 'Accuracy, Precision, Recall, F1-Score',
                documentation: 'Custom estimator HTML file creation'
            },
            results: {
                accuracy: '92%',
                precision: '90%',
                recall: '94%',
                f1Score: '92%'
            },
            applications: [
                'Industrial machinery monitoring',
                'Manufacturing downtime reduction',
                'Equipment lifecycle optimization',
                'Maintenance scheduling automation'
            ],
            futurePlans: [
                'Experiment with Gradient Boosting and XGBoost',
                'Extend to other machinery datasets',
                'Build real-time maintenance prediction dashboard'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/predictive-maintenance-milling',
            demoLink: 'https://predictive-maintenance-demo.streamlit.app',
            tags: ['MLPClassifier', 'Predictive Maintenance', 'SMOTE', 'Industrial IoT', 'Machine Failure Prediction'],
            featured: true,
            projectNumber: 19,
            totalProjects: 120,
            categoryProgress: '19/20 ML Projects'
        },
        {
            id: 'ml-20',
            title: 'Email Spam Detection',
            category: 'Machine Learning',
            domain: 'Natural Language Processing',
            description: 'Developed robust email spam detection system using Logistic Regression and TF-IDF vectorization for binary classification. Applied comprehensive text preprocessing and hyperparameter tuning to protect users from unwanted and malicious emails.',
            image: 'port/ml/20.jpg',
            video: 'port/ml/20.mp4',
            technologies: ['Python', 'Logistic Regression', 'TF-IDF', 'Scikit-learn', 'NLTK', 'Pandas', 'Pickle'],
            frameworks: ['Scikit-learn', 'NLTK'],
            accuracy: '96%',
            modelSize: '8MB',
            trainingTime: '1 hour',
            dataset: 'Email Spam Dataset',
            keyFeatures: [
                'Logistic Regression binary classification',
                'TF-IDF vectorization for text features',
                'Comprehensive text preprocessing pipeline',
                'GridSearchCV hyperparameter optimization',
                'Model serialization with Pickle'
            ],
            technicalDetails: {
                architecture: 'Logistic Regression Classifier',
                preprocessing: 'Stopword removal, HTML tag cleaning, TF-IDF vectorization',
                optimization: 'GridSearchCV for hyperparameter tuning',
                evaluation: 'Accuracy, Precision, Recall, F1-Score, ROC-AUC',
                deployment: 'Pickle model serialization for portability'
            },
            results: {
                accuracy: '96%',
                precision: '95%',
                recall: '97%',
                f1Score: '96%',
                rocAuc: '0.98'
            },
            applications: [
                'Email client spam filtering',
                'Corporate email security systems',
                'Cybersecurity threat detection',
                'Email productivity enhancement'
            ],
            futurePlans: [
                'Experiment with Naive Bayes and XGBoost models',
                'Integrate real-time email stream processing',
                'Extend to phishing detection using metadata'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/email-spam-detection',
            demoLink: 'https://spam-detector-demo.streamlit.app',
            tags: ['Email Spam Detection', 'Logistic Regression', 'TF-IDF', 'NLP', 'Text Classification'],
            featured: true,
            projectNumber: 20,
            totalProjects: 120,
            categoryProgress: '20/20 ML Projects - COMPLETED! 🎉'
        }
    ],

    // 🤖 Deep Learning Projects
    dl: [
        {
            id: 'dl-1',
            title: 'Image Synthesis with DCGAN',
            category: 'Deep Learning',
            domain: 'Generative AI',
            description: 'Implemented Deep Convolutional Generative Adversarial Network (DCGAN) for synthetic image generation. Applied adversarial training between Generator and Discriminator networks to create realistic images from learned data distributions.',
            image: 'port/dl/1.jpg',
            video: 'port/dl/1.mp4',
            technologies: ['Python', 'TensorFlow', 'DCGAN', 'GANs', 'NumPy', 'Matplotlib', 'OpenCV'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '87%',
            modelSize: '65MB',
            trainingTime: '150 epochs (12 hours)',
            dataset: 'Custom Image Dataset',
            keyFeatures: [
                'DCGAN architecture implementation',
                'Adversarial training between Generator and Discriminator',
                'Synthetic image generation from noise vectors',
                'Loss function optimization for stable training',
                'Image quality evaluation metrics'
            ],
            technicalDetails: {
                architecture: 'Deep Convolutional GAN',
                generator: 'Transposed convolutions for upsampling',
                discriminator: 'Convolutional layers for classification',
                training: 'Adversarial loss optimization',
                evaluation: 'Visual inspection and Inception Score'
            },
            results: {
                imageQuality: '87%',
                generatorLoss: 'Converged after 150 epochs',
                discriminatorAccuracy: '85%',
                syntheticImageResolution: '64x64 pixels'
            },
            applications: [
                'Synthetic data generation for training',
                'Creative art and design applications',
                'Data augmentation for limited datasets',
                'Entertainment and gaming content creation'
            ],
            futurePlans: [
                'Experiment with WGAN and StyleGAN architectures',
                'Implement conditional GANs for labeled generation',
                'Build web-based image generation tool'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/image-synthesis-dcgan',
            demoLink: 'https://dcgan-image-generator-demo.streamlit.app',
            tags: ['DCGAN', 'GANs', 'Image Synthesis', 'Generative AI', 'Adversarial Training'],
            featured: true,
            projectNumber: 21,
            totalProjects: 120,
            categoryProgress: '1/20 DL Projects'
        },
        {
            id: 'dl-2',
            title: 'Object Detection using YOLOv4',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented YOLOv4 for real-time object detection with high accuracy using Darknet framework. Applied multi-scale prediction, mosaic data augmentation, and CSPDarknet53 backbone for simultaneous object localization and classification.',
            image: 'port/dl/2.jpg',
            video: 'port/dl/2.mp4',
            technologies: ['Python', 'YOLOv4', 'Darknet', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['Darknet', 'OpenCV'],
            accuracy: '91%',
            modelSize: '245MB',
            trainingTime: '8 hours',
            dataset: 'COCO Dataset / Custom Object Dataset',
            keyFeatures: [
                'YOLOv4 real-time object detection',
                'Multi-scale prediction capability',
                'Mosaic data augmentation technique',
                'CSPDarknet53 backbone architecture',
                'Simultaneous object localization and classification'
            ],
            technicalDetails: {
                architecture: 'YOLOv4 with CSPDarknet53 backbone',
                framework: 'Darknet for YOLO model training',
                augmentation: 'Mosaic data augmentation',
                prediction: 'Multi-scale object detection',
                evaluation: 'mAP (mean Average Precision) and IoU metrics'
            },
            results: {
                accuracy: '91%',
                mAP: '0.88',
                inferenceSpeed: '45 FPS',
                iouThreshold: '0.5'
            },
            applications: [
                'Real-time surveillance systems',
                'Autonomous vehicle object detection',
                'Traffic monitoring and analysis',
                'Security and safety applications'
            ],
            futurePlans: [
                'Implement YOLOv5 and YOLOv7 for comparison',
                'Develop video-based object detection system',
                'Integrate with webcam/drone for live detection'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/yolov4-object-detection',
            demoLink: 'https://yolov4-detection-demo.streamlit.app',
            tags: ['YOLOv4', 'Object Detection', 'Darknet', 'Real-time Processing', 'Computer Vision'],
            featured: true,
            projectNumber: 22,
            totalProjects: 120,
            categoryProgress: '2/20 DL Projects'
        },
        {
            id: 'dl-3',
            title: 'Style Transfer using cGAN (Horse to Zebra)',
            category: 'Deep Learning',
            domain: 'Generative AI',
            description: 'Implemented Conditional Generative Adversarial Network (cGAN) for style transfer, transforming horse images into zebra-style outputs. Applied adversarial loss and cycle-consistency loss for visually accurate domain mapping.',
            image: 'port/dl/3.jpg',
            video: 'port/dl/3.mp4',
            technologies: ['Python', 'TensorFlow', 'cGAN', 'CycleGAN', 'NumPy', 'Matplotlib', 'GPU'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '89%',
            modelSize: '120MB',
            trainingTime: '10 hours',
            dataset: 'Horse to Zebra Dataset (CycleGAN)',
            keyFeatures: [
                'Conditional GAN for domain-to-domain mapping',
                'Adversarial loss for realistic generation',
                'Cycle-consistency loss for content preservation',
                'GPU-accelerated training optimization',
                'Visual style transfer with structural integrity'
            ],
            technicalDetails: {
                architecture: 'Conditional GAN with Generator-Discriminator',
                lossFunction: 'Adversarial Loss + Cycle-Consistency Loss',
                training: 'Domain mapping from horses to zebras',
                optimization: 'GPU acceleration for computational efficiency',
                evaluation: 'Visual quality assessment and structural consistency'
            },
            results: {
                styleTransferQuality: '89%',
                structuralIntegrity: '92%',
                visualRealism: '87%',
                processingTime: '2.5 seconds per image'
            },
            applications: [
                'Artistic style transformation',
                'Domain adaptation for computer vision',
                'Creative content generation',
                'Image-to-image translation tasks'
            ],
            futurePlans: [
                'Explore other domain pairs (day-to-night, summer-to-winter)',
                'Implement advanced GAN architectures (StyleGAN, BigGAN)',
                'Build interactive style transfer web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/style-transfer-cgan',
            demoLink: 'https://horse-zebra-style-transfer-demo.streamlit.app',
            tags: ['cGAN', 'Style Transfer', 'CycleGAN', 'Domain Mapping', 'Generative AI'],
            featured: true,
            projectNumber: 23,
            totalProjects: 120,
            categoryProgress: '3/20 DL Projects'
        },
        {
            id: 'dl-4',
            title: 'Sequence-to-Sequence Machine Translation',
            category: 'Deep Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented machine translation using two approaches: Seq2Seq with LSTM for French-to-English translation and Transformer architecture for English-to-Hindi translation. Applied teacher forcing, attention mechanisms, and custom tokenizers.',
            image: 'port/dl/4.jpg',
            video: 'port/dl/4.mp4',
            technologies: ['Python', 'TensorFlow', 'Transformers', 'LSTM', 'Hugging Face', 'NLTK', 'Tokenizers'],
            frameworks: ['TensorFlow', 'Hugging Face Transformers'],
            accuracy: '85%',
            modelSize: '180MB',
            trainingTime: '6 hours',
            dataset: 'French-English & English-Hindi Parallel Datasets',
            keyFeatures: [
                'Seq2Seq LSTM encoder-decoder architecture',
                'Transformer model with self-attention mechanisms',
                'Teacher forcing for efficient training',
                'Multi-head attention for long-range dependencies',
                'Custom tokenizers for multilingual preprocessing'
            ],
            technicalDetails: {
                seq2seqArchitecture: 'LSTM Encoder-Decoder with attention',
                transformerArchitecture: 'Multi-head self-attention with positional encoding',
                training: 'Teacher forcing and masked language modeling',
                preprocessing: 'Custom tokenization and sequence padding',
                evaluation: 'BLEU score and translation quality assessment'
            },
            results: {
                frenchEnglishAccuracy: '83%',
                englishHindiAccuracy: '87%',
                bleuScore: '0.72',
                translationFluency: '85%'
            },
            applications: [
                'Real-time language translation services',
                'Cross-lingual communication platforms',
                'Multilingual content localization',
                'Educational language learning tools'
            ],
            futurePlans: [
                'Implement zero-shot translation capabilities',
                'Explore multilingual BERT for translation',
                'Build real-time translation web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/seq2seq-machine-translation',
            demoLink: 'https://machine-translation-demo.streamlit.app',
            tags: ['Seq2Seq', 'Machine Translation', 'LSTM', 'Transformers', 'Multilingual NLP'],
            featured: true,
            projectNumber: 24,
            totalProjects: 120,
            categoryProgress: '4/20 DL Projects'
        },
        {
            id: 'dl-5',
            title: 'U-Net for Image Segmentation',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented U-Net fully convolutional network for precise pixel-wise image segmentation. Applied to medical image segmentation for tumor detection and general object segmentation with encoder-decoder architecture and skip connections.',
            image: 'port/dl/5.jpg',
            video: 'port/dl/5.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'U-Net', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '92%',
            modelSize: '85MB',
            trainingTime: '5 hours',
            dataset: 'Medical Image Segmentation Dataset',
            keyFeatures: [
                'U-Net encoder-decoder architecture',
                'Skip connections for spatial detail preservation',
                'Pixel-wise segmentation capability',
                'Data augmentation for enhanced performance',
                'Medical image tumor detection application'
            ],
            technicalDetails: {
                architecture: 'U-Net with encoder-decoder and skip connections',
                lossFunction: 'Binary Cross-Entropy with Dice Loss',
                optimizer: 'Adam',
                preprocessing: 'Image normalization and data augmentation',
                evaluation: 'IoU, Dice Coefficient, Pixel Accuracy'
            },
            results: {
                accuracy: '92%',
                iouScore: '0.89',
                diceCoefficient: '0.91',
                pixelAccuracy: '94%'
            },
            applications: [
                'Medical image analysis and diagnosis',
                'Tumor detection and segmentation',
                'Autonomous vehicle perception',
                'Satellite image analysis'
            ],
            futurePlans: [
                'Implement U-Net++ and Attention U-Net variants',
                'Apply to 3D medical image segmentation',
                'Deploy for real-time medical diagnosis support'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/unet-image-segmentation',
            demoLink: 'https://unet-segmentation-demo.streamlit.app',
            tags: ['U-Net', 'Image Segmentation', 'Medical Imaging', 'Encoder-Decoder', 'Skip Connections'],
            featured: true,
            projectNumber: 25,
            totalProjects: 120,
            categoryProgress: '5/20 DL Projects'
        },
        {
            id: 'dl-6',
            title: 'Text Generation with GPT-2',
            category: 'Deep Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented text generation using OpenAI GPT-2 transformer model with fine-tuning capabilities. Applied Hugging Face Transformers for coherent, context-aware text generation including chatbot responses, story generation, and content creation.',
            image: 'port/dl/6.jpg',
            video: 'port/dl/6.mp4',
            technologies: ['Python', 'GPT-2', 'Transformers', 'Hugging Face', 'PyTorch', 'NLTK', 'Tokenizers'],
            frameworks: ['Hugging Face Transformers', 'PyTorch'],
            accuracy: '88%',
            modelSize: '550MB',
            trainingTime: '4 hours',
            dataset: 'Custom Text Dataset + OpenWebText',
            keyFeatures: [
                'GPT-2 transformer-based text generation',
                'Fine-tuning on custom datasets',
                'Context-aware coherent text production',
                'Multi-purpose applications (chatbot, stories, content)',
                'Hugging Face Transformers integration'
            ],
            technicalDetails: {
                architecture: 'GPT-2 Transformer with self-attention',
                finetuning: 'Domain-specific dataset adaptation',
                generation: 'Autoregressive text generation',
                preprocessing: 'Tokenization and sequence formatting',
                evaluation: 'Perplexity and human evaluation metrics'
            },
            results: {
                textQuality: '88%',
                perplexity: '23.5',
                coherenceScore: '85%',
                contextRelevance: '90%'
            },
            applications: [
                'AI-powered chatbot systems',
                'Creative story and content generation',
                'Automated writing assistance',
                'Personalized text completion'
            ],
            futurePlans: [
                'Implement GPT-3 and GPT-4 for comparison',
                'Develop specialized domain fine-tuning',
                'Build interactive text generation web app'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/gpt2-text-generation',
            demoLink: 'https://gpt2-text-generator-demo.streamlit.app',
            tags: ['GPT-2', 'Text Generation', 'Transformers', 'Fine-tuning', 'Language Models'],
            featured: true,
            projectNumber: 26,
            totalProjects: 120,
            categoryProgress: '6/20 DL Projects'
        },
        {
            id: 'dl-7',
            title: 'Neural Style Transfer for Image Transformation',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented Neural Style Transfer using TensorFlow Hub pretrained model for fast artistic image transformation. Applied style transfer to multiple content images with optimized balance between content and style preservation.',
            image: 'port/dl/7.jpg',
            video: 'port/dl/7.mp4',
            technologies: ['Python', 'TensorFlow', 'TensorFlow Hub', 'OpenCV', 'NumPy', 'Matplotlib', 'PIL'],
            frameworks: ['TensorFlow', 'TensorFlow Hub'],
            accuracy: '93%',
            modelSize: '26MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'Custom Content and Style Images',
            keyFeatures: [
                'TensorFlow Hub pretrained NST model',
                'Fast style transfer implementation',
                'Content and style preservation optimization',
                'Real-time artistic filter application',
                'Multiple artistic style support'
            ],
            technicalDetails: {
                architecture: 'Pretrained Fast Neural Style Transfer',
                model: 'TensorFlow Hub NST model',
                optimization: 'Content-style balance tuning',
                preprocessing: 'Image normalization and resizing',
                evaluation: 'Visual quality and style similarity assessment'
            },
            results: {
                styleTransferQuality: '93%',
                contentPreservation: '89%',
                processingSpeed: '0.8 seconds per image',
                styleSimilarity: '91%'
            },
            applications: [
                'AI-powered artistic filters',
                'Real-time style transfer applications',
                'Creative design and art generation',
                'Social media content enhancement'
            ],
            futurePlans: [
                'Implement custom NST training from scratch',
                'Develop mobile app for real-time style transfer',
                'Explore video style transfer capabilities'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/neural-style-transfer',
            demoLink: 'https://neural-style-transfer-demo.streamlit.app',
            tags: ['Neural Style Transfer', 'TensorFlow Hub', 'Artistic Filters', 'Image Transformation', 'Computer Vision'],
            featured: true,
            projectNumber: 27,
            totalProjects: 120,
            categoryProgress: '7/20 DL Projects'
        },
        {
            id: 'dl-8',
            title: 'Deepfake Image Generation Using ProGAN',
            category: 'Deep Learning',
            domain: 'Generative AI',
            description: 'Implemented deepfake image generation using Progressive Growing GAN (ProGAN) for high-resolution synthetic face creation. Applied progressive training methodology to generate realistic human faces with smooth transitions and ethical considerations.',
            image: 'port/dl/8.jpg',
            video: 'port/dl/8.mp4',
            technologies: ['Python', 'TensorFlow', 'ProGAN', 'Progressive GAN', 'NumPy', 'PIL', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '87%',
            modelSize: '180MB',
            trainingTime: 'Pretrained (Progressive Training)',
            dataset: 'High-Resolution Face Dataset',
            keyFeatures: [
                'Progressive Growing GAN architecture',
                'High-resolution synthetic face generation',
                'Smooth image transition capabilities',
                'Pretrained model utilization',
                'Ethical deepfake technology exploration'
            ],
            technicalDetails: {
                architecture: 'Progressive Growing GAN',
                training: 'Progressive resolution increase (4x4 to 1024x1024)',
                generation: 'Latent space interpolation for face synthesis',
                stabilization: 'Progressive training for stable GAN convergence',
                evaluation: 'FID score and visual quality assessment'
            },
            results: {
                imageQuality: '87%',
                resolution: '1024x1024 pixels',
                fidScore: '15.2',
                generationTime: '3.2 seconds per image'
            },
            applications: [
                'Synthetic dataset generation for training',
                'Entertainment and media production',
                'Privacy-preserving face anonymization',
                'Research in generative modeling'
            ],
            futurePlans: [
                'Explore StyleGAN and StyleGAN2 architectures',
                'Implement deepfake detection systems',
                'Study ethical implications and responsible AI'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/deepfake-progan',
            demoLink: 'https://deepfake-generation-demo.streamlit.app',
            tags: ['ProGAN', 'Deepfake Generation', 'Progressive GAN', 'Synthetic Faces', 'Generative AI'],
            featured: true,
            projectNumber: 28,
            totalProjects: 120,
            categoryProgress: '8/20 DL Projects'
        },
        {
            id: 'dl-9',
            title: 'Predictive Analysis Using RNN, LSTM & GRU (IMDB Sentiment Analysis)',
            category: 'Deep Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented sentiment analysis on IMDB movie reviews using three RNN variants: basic RNN, LSTM, and GRU. Compared model performance for binary classification with comprehensive preprocessing and evaluation metrics.',
            image: 'port/dl/9.jpg',
            video: 'port/dl/9.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'RNN', 'LSTM', 'GRU', 'NLTK', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '89%',
            modelSize: '45MB',
            trainingTime: '3 hours',
            dataset: 'IMDB Movie Reviews Dataset',
            keyFeatures: [
                'Three RNN variant implementations (RNN, LSTM, GRU)',
                'Sequential data processing for sentiment analysis',
                'Long-term dependency handling comparison',
                'Comprehensive text preprocessing pipeline',
                'Model performance comparison and evaluation'
            ],
            technicalDetails: {
                architectures: 'RNN, LSTM, GRU with embedding layers',
                preprocessing: 'Tokenization, sequence padding, embedding',
                training: 'Binary classification with dropout regularization',
                evaluation: 'Accuracy, loss, precision, recall comparison',
                optimization: 'Adam optimizer with learning rate scheduling'
            },
            results: {
                rnnAccuracy: '82%',
                lstmAccuracy: '89%',
                gruAccuracy: '87%',
                bestModel: 'LSTM with 89% accuracy'
            },
            applications: [
                'Social media sentiment monitoring',
                'Customer feedback analysis systems',
                'Product review classification',
                'Brand sentiment tracking'
            ],
            futurePlans: [
                'Implement Bidirectional LSTM and GRU',
                'Explore Transformer-based models (BERT)',
                'Deploy real-time sentiment analysis API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/rnn-lstm-gru-sentiment-analysis',
            demoLink: 'https://imdb-sentiment-analysis-demo.streamlit.app',
            tags: ['RNN', 'LSTM', 'GRU', 'Sentiment Analysis', 'IMDB', 'Sequential Models'],
            featured: true,
            projectNumber: 29,
            totalProjects: 120,
            categoryProgress: '9/20 DL Projects'
        },
        {
            id: 'dl-10',
            title: 'Image Captioning Using VGG & LSTM',
            category: 'Deep Learning',
            domain: 'Computer Vision + NLP',
            description: 'Built AI-powered image captioning system combining VGG16 CNN for feature extraction and LSTM for text generation. Applied multimodal deep learning to generate meaningful captions for images using Flickr8k dataset.',
            image: 'port/dl/10.jpg',
            video: 'port/dl/10.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'VGG16', 'LSTM', 'OpenCV', 'NLTK', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '84%',
            modelSize: '95MB',
            trainingTime: '6 hours',
            dataset: 'Flickr8k Dataset',
            keyFeatures: [
                'VGG16 pretrained CNN for image feature extraction',
                'LSTM-based text generation decoder',
                'Multimodal deep learning architecture',
                'Word embedding layer for text representation',
                'Automated caption generation for images'
            ],
            technicalDetails: {
                encoder: 'VGG16 CNN for visual feature extraction',
                decoder: 'LSTM for sequential text generation',
                preprocessing: 'Image normalization and text tokenization',
                training: 'Teacher forcing with cross-entropy loss',
                evaluation: 'BLEU score and caption quality assessment'
            },
            results: {
                accuracy: '84%',
                bleuScore: '0.67',
                captionQuality: '82%',
                vocabularySize: '8000+ words'
            },
            applications: [
                'AI-powered photo description systems',
                'Accessibility tools for visually impaired',
                'Social media content automation',
                'Educational image analysis tools'
            ],
            futurePlans: [
                'Implement attention mechanisms for better focus',
                'Explore Transformer-based image captioning',
                'Deploy real-time captioning web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/image-captioning-vgg-lstm',
            demoLink: 'https://image-captioning-demo.streamlit.app',
            tags: ['Image Captioning', 'VGG16', 'LSTM', 'Multimodal AI', 'Computer Vision + NLP'],
            featured: true,
            projectNumber: 30,
            totalProjects: 120,
            categoryProgress: '10/20 DL Projects'
        },
        {
            id: 'dl-11',
            title: 'Mask Detection Using MobileNet',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented real-time face mask detection system using MobileNetV2 lightweight CNN architecture. Applied transfer learning with custom classification head for binary classification of masked vs non-masked faces with OpenCV integration.',
            image: 'port/dl/11.jpg',
            video: 'port/dl/11.mp4',
            technologies: ['Python', 'TensorFlow', 'MobileNetV2', 'OpenCV', 'Transfer Learning', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras', 'OpenCV'],
            accuracy: '96%',
            modelSize: '14MB',
            trainingTime: '2 hours',
            dataset: 'Face Mask Dataset (Kaggle)',
            keyFeatures: [
                'MobileNetV2 lightweight CNN architecture',
                'Transfer learning with pretrained weights',
                'Custom classification head for binary classification',
                'Real-time detection with OpenCV integration',
                'Efficient mobile-friendly model deployment'
            ],
            technicalDetails: {
                architecture: 'MobileNetV2 + Custom Dense Layers',
                transferLearning: 'Pretrained ImageNet weights fine-tuning',
                preprocessing: 'Image resize, normalization, augmentation',
                classification: 'Binary classification (Masked/No Mask)',
                deployment: 'Real-time OpenCV video processing'
            },
            results: {
                accuracy: '96%',
                precision: '95%',
                recall: '97%',
                f1Score: '96%',
                inferenceSpeed: '30 FPS'
            },
            applications: [
                'Public health monitoring systems',
                'COVID-19 safety compliance checking',
                'Smart surveillance for mask enforcement',
                'Mobile health screening applications'
            ],
            futurePlans: [
                'Deploy on mobile devices using TensorFlow Lite',
                'Extend to detect mask types and quality',
                'Integrate with IoT systems for automated alerts'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/mask-detection-mobilenet',
            demoLink: 'https://mask-detection-demo.streamlit.app',
            tags: ['MobileNet', 'Mask Detection', 'Transfer Learning', 'Real-time Processing', 'Public Health'],
            featured: true,
            projectNumber: 31,
            totalProjects: 120,
            categoryProgress: '11/20 DL Projects'
        },
        {
            id: 'dl-12',
            title: 'Human Pose Estimation Using OpenPose',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented real-time human pose estimation using OpenPose pretrained model for detecting body keypoints and generating pose skeletons. Applied multi-person pose detection with OpenCV integration for sports analytics and motion tracking.',
            image: 'port/dl/12.jpg',
            video: 'port/dl/12.mp4',
            technologies: ['Python', 'OpenPose', 'OpenCV', 'TensorFlow', 'NumPy', 'Matplotlib'],
            frameworks: ['OpenPose', 'OpenCV', 'TensorFlow'],
            accuracy: '91%',
            modelSize: '200MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'COCO Keypoint Dataset',
            keyFeatures: [
                'OpenPose pretrained model for pose detection',
                'Real-time body keypoint extraction',
                'Multi-person pose estimation capability',
                'Pose skeleton generation and visualization',
                '2D keypoint detection for 18 body joints'
            ],
            technicalDetails: {
                architecture: 'OpenPose CNN with Part Affinity Fields',
                keypoints: '18 body joints detection (head, shoulders, arms, legs)',
                processing: 'Real-time video stream analysis',
                multiPerson: 'Simultaneous pose detection for multiple people',
                evaluation: 'Keypoint accuracy and pose skeleton quality'
            },
            results: {
                accuracy: '91%',
                keypointPrecision: '89%',
                processingSpeed: '25 FPS',
                multiPersonSupport: 'Up to 10 people simultaneously'
            },
            applications: [
                'Sports analytics and performance tracking',
                'AI fitness and exercise monitoring',
                'Motion capture for animation',
                'Surveillance and behavior analysis'
            ],
            futurePlans: [
                'Implement 3D pose estimation capabilities',
                'Develop action recognition from pose sequences',
                'Create fitness coaching application with pose feedback'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/human-pose-estimation-openpose',
            demoLink: 'https://pose-estimation-demo.streamlit.app',
            tags: ['OpenPose', 'Pose Estimation', 'Keypoint Detection', 'Motion Tracking', 'Sports Analytics'],
            featured: true,
            projectNumber: 32,
            totalProjects: 120,
            categoryProgress: '12/20 DL Projects'
        },
        {
            id: 'dl-13',
            title: 'Voice Synthesis Using Tacotron2 & HiFi-GAN',
            category: 'Deep Learning',
            domain: 'Audio Processing + NLP',
            description: 'Implemented end-to-end text-to-speech synthesis using Tacotron2 for mel-spectrogram generation and HiFi-GAN for high-quality waveform synthesis. Applied deep learning for realistic voice generation from text input.',
            image: 'port/dl/13.jpg',
            video: 'port/dl/13.mp4',
            technologies: ['Python', 'PyTorch', 'TensorFlow', 'Tacotron2', 'HiFi-GAN', 'LibROSA', 'NumPy'],
            frameworks: ['PyTorch', 'TensorFlow'],
            accuracy: '88%',
            modelSize: '150MB',
            trainingTime: '15 hours',
            dataset: 'LJSpeech Dataset',
            keyFeatures: [
                'Tacotron2 for text-to-mel-spectrogram conversion',
                'HiFi-GAN for mel-to-waveform synthesis',
                'End-to-end speech synthesis pipeline',
                'High-quality natural voice generation',
                'Custom dataset fine-tuning capability'
            ],
            technicalDetails: {
                tacotron2: 'Sequence-to-sequence model for mel-spectrogram generation',
                hifiGAN: 'Generative adversarial network for waveform synthesis',
                pipeline: 'Text → Mel-Spectrogram → Audio Waveform',
                training: 'Multi-stage training with attention mechanisms',
                evaluation: 'MOS (Mean Opinion Score) and audio quality metrics'
            },
            results: {
                voiceQuality: '88%',
                mosScore: '4.2/5.0',
                synthesisSpeed: '0.8x real-time',
                naturalness: '85%'
            },
            applications: [
                'AI voice assistants and chatbots',
                'Audiobook and podcast generation',
                'Accessibility tools for text-to-speech',
                'Personalized voice synthesis systems'
            ],
            futurePlans: [
                'Implement multi-speaker voice cloning',
                'Explore FastSpeech2 for faster synthesis',
                'Develop real-time TTS web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/voice-synthesis-tacotron2-hifigan',
            demoLink: 'https://voice-synthesis-demo.streamlit.app',
            tags: ['Tacotron2', 'HiFi-GAN', 'Text-to-Speech', 'Voice Synthesis', 'Audio Processing'],
            featured: true,
            projectNumber: 33,
            totalProjects: 120,
            categoryProgress: '13/20 DL Projects'
        },
        {
            id: 'dl-14',
            title: 'Super-Resolution for Image Reconstruction using GANs',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented image super-resolution using SRGAN to enhance low-resolution images with high-quality reconstruction. Applied adversarial training with perceptual loss and VGG-19 feature extraction for superior upscaling results.',
            image: 'port/dl/14.jpg',
            video: 'port/dl/14.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'SRGAN', 'VGG-19', 'OpenCV', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '90%',
            modelSize: '120MB',
            trainingTime: '8 hours',
            dataset: 'DIV2K High-Quality Image Dataset',
            keyFeatures: [
                'SRGAN for adversarial super-resolution training',
                'Generator-Discriminator architecture for image enhancement',
                'Perceptual loss with VGG-19 feature extraction',
                'High-quality upscaling from low-resolution inputs',
                'Fine detail reconstruction capability'
            ],
            technicalDetails: {
                architecture: 'SRGAN with ResNet-based Generator',
                lossFunction: 'Adversarial Loss + Perceptual Loss + Content Loss',
                featureExtraction: 'Pre-trained VGG-19 for perceptual similarity',
                upscaling: '4x resolution enhancement (LR to HR)',
                evaluation: 'PSNR, SSIM, and visual quality assessment'
            },
            results: {
                imageQuality: '90%',
                psnrScore: '28.5 dB',
                ssimScore: '0.85',
                upscalingFactor: '4x resolution enhancement'
            },
            applications: [
                'Medical imaging enhancement',
                'Satellite imagery super-resolution',
                'Video quality enhancement',
                'Photography and digital art improvement'
            ],
            futurePlans: [
                'Implement ESRGAN for better quality',
                'Explore Real-ESRGAN for real-world images',
                'Deploy real-time super-resolution web service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/super-resolution-srgan',
            demoLink: 'https://super-resolution-demo.streamlit.app',
            tags: ['SRGAN', 'Super-Resolution', 'Image Enhancement', 'GANs', 'Perceptual Loss'],
            featured: true,
            projectNumber: 34,
            totalProjects: 120,
            categoryProgress: '14/20 DL Projects'
        },
        {
            id: 'dl-15',
            title: 'Brain Tumor Detection using VGG19',
            category: 'Deep Learning',
            domain: 'Medical AI',
            description: 'Implemented brain tumor detection from MRI scans using pretrained VGG19 with transfer learning. Applied fine-tuning, data augmentation, and medical image preprocessing for accurate tumor classification in healthcare diagnostics.',
            image: 'port/dl/15.jpg',
            video: 'port/dl/15.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'VGG19', 'Transfer Learning', 'OpenCV', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '94%',
            modelSize: '528MB',
            trainingTime: '4 hours',
            dataset: 'Brain MRI Dataset (Tumor vs No Tumor)',
            keyFeatures: [
                'VGG19 pretrained model with transfer learning',
                'Fine-tuning for medical image classification',
                'Data augmentation for improved generalization',
                'Medical image preprocessing pipeline',
                'Binary classification for tumor detection'
            ],
            technicalDetails: {
                architecture: 'VGG19 + Custom Classification Head',
                transferLearning: 'ImageNet pretrained weights fine-tuning',
                preprocessing: 'MRI image normalization and augmentation',
                classification: 'Binary classification (Tumor/No Tumor)',
                evaluation: 'Accuracy, Precision, Recall, F1-Score'
            },
            results: {
                accuracy: '94%',
                precision: '93%',
                recall: '95%',
                f1Score: '94%',
                sensitivity: '95%'
            },
            applications: [
                'Medical diagnostic assistance systems',
                'Radiology screening automation',
                'Early tumor detection tools',
                'Healthcare AI decision support'
            ],
            futurePlans: [
                'Implement 3D CNN for volumetric MRI analysis',
                'Explore ResNet and EfficientNet architectures',
                'Deploy as medical diagnostic web application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/brain-tumor-detection-vgg19',
            demoLink: 'https://brain-tumor-detection-demo.streamlit.app',
            tags: ['VGG19', 'Brain Tumor Detection', 'Medical AI', 'Transfer Learning', 'MRI Analysis'],
            featured: true,
            projectNumber: 35,
            totalProjects: 120,
            categoryProgress: '15/20 DL Projects'
        },
        {
            id: 'dl-16',
            title: 'Emotion Detection from Text using BERT',
            category: 'Deep Learning',
            domain: 'Natural Language Processing',
            description: 'Implemented emotion classification from text using fine-tuned BERT transformer model. Applied Hugging Face Transformers with custom training pipeline for multi-class emotion detection including happy, sad, angry, and other emotional states.',
            image: 'port/dl/16.jpg',
            video: 'port/dl/16.mp4',
            technologies: ['Python', 'BERT', 'Transformers', 'Hugging Face', 'PyTorch', 'NLTK', 'Pandas'],
            frameworks: ['Hugging Face Transformers', 'PyTorch'],
            accuracy: '93%',
            modelSize: '440MB',
            trainingTime: '3 hours',
            dataset: 'Emotion Classification Dataset',
            keyFeatures: [
                'BERT transformer fine-tuning for emotion classification',
                'Hugging Face Trainer API for efficient training',
                'Multi-class emotion detection (Happy, Sad, Angry, etc.)',
                'BERT tokenizer for text preprocessing',
                'Contextual understanding of emotional expressions'
            ],
            technicalDetails: {
                architecture: 'BERT + Classification Head',
                finetuning: 'Transfer learning with emotion-specific dataset',
                tokenization: 'BERT WordPiece tokenization',
                classification: 'Multi-class emotion classification',
                evaluation: 'Accuracy, F1-Score, Confusion Matrix'
            },
            results: {
                accuracy: '93%',
                f1Score: '92%',
                precision: '93%',
                recall: '92%',
                emotionClasses: '6 emotion categories'
            },
            applications: [
                'Social media sentiment monitoring',
                'Customer feedback emotion analysis',
                'Mental health assessment tools',
                'Chatbot emotional intelligence'
            ],
            futurePlans: [
                'Implement RoBERTa and DistilBERT comparison',
                'Explore multilingual emotion detection',
                'Deploy real-time emotion analysis API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/emotion-detection-bert',
            demoLink: 'https://emotion-detection-demo.streamlit.app',
            tags: ['BERT', 'Emotion Detection', 'Text Classification', 'Transformers', 'NLP'],
            featured: true,
            projectNumber: 36,
            totalProjects: 120,
            categoryProgress: '16/20 DL Projects'
        },
        {
            id: 'dl-17',
            title: 'Speech Recognition using Librosa',
            category: 'Deep Learning',
            domain: 'Audio Processing + NLP',
            description: 'Built speech recognition system using Librosa for audio feature extraction and deep learning models for speech-to-text conversion. Applied MFCC, mel-spectrograms, and CNN/RNN architectures for accurate speech recognition.',
            image: 'port/dl/17.jpg',
            video: 'port/dl/17.mp4',
            technologies: ['Python', 'Librosa', 'TensorFlow', 'Keras', 'PyTorch', 'MFCC', 'LSTM', 'CNN'],
            frameworks: ['Librosa', 'TensorFlow', 'PyTorch'],
            accuracy: '86%',
            modelSize: '75MB',
            trainingTime: '5 hours',
            dataset: 'Common Voice Dataset / Custom Audio Data',
            keyFeatures: [
                'Librosa for comprehensive audio preprocessing',
                'MFCC and mel-spectrogram feature extraction',
                'CNN/RNN hybrid architecture for speech recognition',
                'Audio data augmentation for robustness',
                'Real-time speech-to-text conversion'
            ],
            technicalDetails: {
                audioProcessing: 'Librosa for resampling, noise reduction, normalization',
                featureExtraction: 'MFCC, Mel-spectrograms, Chroma features',
                architecture: 'CNN for feature learning + LSTM for sequence modeling',
                augmentation: 'Time stretching, pitch shifting, noise addition',
                evaluation: 'Word Error Rate (WER) and Character Error Rate (CER)'
            },
            results: {
                accuracy: '86%',
                wordErrorRate: '14%',
                characterErrorRate: '8%',
                processingSpeed: '1.2x real-time'
            },
            applications: [
                'Voice assistants and smart speakers',
                'Automated transcription services',
                'Accessibility tools for hearing impaired',
                'Voice-controlled applications'
            ],
            futurePlans: [
                'Implement Transformer-based speech recognition',
                'Explore Wav2Vec2 and Whisper models',
                'Deploy real-time speech recognition web app'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/speech-recognition-librosa',
            demoLink: 'https://speech-recognition-demo.streamlit.app',
            tags: ['Speech Recognition', 'Librosa', 'MFCC', 'Audio Processing', 'LSTM'],
            featured: true,
            projectNumber: 37,
            totalProjects: 120,
            categoryProgress: '17/20 DL Projects'
        },
        {
            id: 'dl-18',
            title: 'AI-Based Cybersecurity Threat Detection using LSTM',
            category: 'Deep Learning',
            domain: 'Cybersecurity',
            description: 'Implemented AI-powered cybersecurity threat detection system using LSTM for sequential network traffic analysis. Applied feature engineering, SMOTE for class imbalance, and achieved 99% accuracy in detecting various cyber attacks.',
            image: 'port/dl/18.jpg',
            video: 'port/dl/18.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'LSTM', 'Scikit-learn', 'SMOTE', 'Pandas', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras', 'Scikit-learn'],
            accuracy: '99%',
            modelSize: '35MB',
            trainingTime: '4 hours',
            dataset: 'CIC-IDS2017 Intrusion Detection Dataset',
            keyFeatures: [
                'LSTM for sequential network traffic pattern analysis',
                'Multi-class cyber attack classification',
                'SMOTE for handling class imbalance',
                'Real-time threat detection capability',
                'Feature engineering for network security metrics'
            ],
            technicalDetails: {
                architecture: 'LSTM with Dense layers for classification',
                preprocessing: 'Feature scaling, normalization, SMOTE oversampling',
                attackTypes: 'DDoS, Botnet, Brute Force, Web Attacks, Infiltration',
                evaluation: 'Accuracy, Precision, Recall, F1-Score, Confusion Matrix',
                deployment: 'Real-time network traffic monitoring'
            },
            results: {
                accuracy: '99%',
                precision: '98%',
                recall: '99%',
                f1Score: '99%',
                falsePositiveRate: '0.8%'
            },
            applications: [
                'Network intrusion detection systems',
                'Real-time cybersecurity monitoring',
                'Enterprise security threat analysis',
                'Automated incident response systems'
            ],
            futurePlans: [
                'Implement Transformer-based threat detection',
                'Explore federated learning for distributed security',
                'Deploy as enterprise cybersecurity solution'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/cybersecurity-threat-detection-lstm',
            demoLink: 'https://cyber-threat-detection-demo.streamlit.app',
            tags: ['Cybersecurity', 'LSTM', 'Threat Detection', 'Network Security', 'Anomaly Detection'],
            featured: true,
            projectNumber: 38,
            totalProjects: 120,
            categoryProgress: '18/20 DL Projects'
        },
        {
            id: 'dl-19',
            title: 'Pothole Detection using YOLOv11 Nano',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented real-time pothole detection system using YOLOv11 Nano for smart transportation and road safety. Applied custom dataset training with YOLO format annotations and optimized for edge device deployment.',
            image: 'port/dl/19.jpg',
            video: 'port/dl/19.mp4',
            technologies: ['Python', 'YOLOv11', 'PyTorch', 'OpenCV', 'Ultralytics', 'NumPy', 'Matplotlib'],
            frameworks: ['YOLOv11', 'PyTorch', 'OpenCV'],
            accuracy: '92%',
            modelSize: '6MB',
            trainingTime: '3 hours',
            dataset: 'Custom Pothole Dataset (YOLO Format)',
            keyFeatures: [
                'YOLOv11 Nano lightweight architecture',
                'Real-time pothole detection capability',
                'Custom dataset training with YOLO annotations',
                'Edge device optimization (Raspberry Pi, Jetson Nano)',
                'Multi-lighting condition robustness'
            ],
            technicalDetails: {
                architecture: 'YOLOv11 Nano for efficient object detection',
                optimization: 'Lightweight model for edge deployment',
                training: 'Custom dataset with data augmentation',
                inference: 'Real-time detection with high FPS',
                evaluation: 'mAP, Precision, Recall, IoU metrics'
            },
            results: {
                accuracy: '92%',
                mAP: '0.89',
                inferenceSpeed: '60 FPS',
                modelSize: '6MB (optimized for edge)'
            },
            applications: [
                'Smart vehicle road safety systems',
                'Infrastructure maintenance automation',
                'Autonomous vehicle perception',
                'Municipal road monitoring systems'
            ],
            futurePlans: [
                'Deploy on mobile devices with real-time alerts',
                'Integrate with GPS for pothole mapping',
                'Expand to detect other road hazards'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/pothole-detection-yolov11',
            demoLink: 'https://pothole-detection-demo.streamlit.app',
            tags: ['YOLOv11', 'Pothole Detection', 'Road Safety', 'Edge Computing', 'Smart Transportation'],
            featured: true,
            projectNumber: 39,
            totalProjects: 120,
            categoryProgress: '19/20 DL Projects'
        },
        {
            id: 'dl-20',
            title: 'License Plate Detection using YOLOv9',
            category: 'Deep Learning',
            domain: 'Computer Vision',
            description: 'Implemented real-time license plate detection system using YOLOv9 for automatic vehicle recognition and smart surveillance. Applied Roboflow dataset with fine-tuning for various vehicle types and lighting conditions.',
            image: 'port/dl/20.jpg',
            video: 'port/dl/20.mp4',
            technologies: ['Python', 'YOLOv9', 'PyTorch', 'OpenCV', 'Roboflow', 'NumPy', 'Matplotlib'],
            frameworks: ['YOLOv9', 'PyTorch', 'OpenCV'],
            accuracy: '94%',
            modelSize: '76MB',
            trainingTime: '4 hours',
            dataset: 'License Plate Dataset (Roboflow)',
            keyFeatures: [
                'YOLOv9 latest architecture for object detection',
                'Real-time license plate recognition capability',
                'Multi-vehicle type and lighting condition support',
                'High-speed inference for traffic monitoring',
                'Integration ready for ANPR systems'
            ],
            technicalDetails: {
                architecture: 'YOLOv9 with advanced feature extraction',
                training: 'Fine-tuning on Roboflow license plate dataset',
                optimization: 'Real-time inference with GPU acceleration',
                robustness: 'Various plate sizes, angles, and lighting conditions',
                evaluation: 'mAP, Precision, Recall, Detection Speed'
            },
            results: {
                accuracy: '94%',
                mAP: '0.91',
                inferenceSpeed: '50 FPS',
                detectionPrecision: '93%'
            },
            applications: [
                'Automatic Number Plate Recognition (ANPR)',
                'Traffic monitoring and law enforcement',
                'Toll booth automation systems',
                'Smart parking management'
            ],
            futurePlans: [
                'Integrate OCR for complete plate text extraction',
                'Deploy on edge devices for real-time monitoring',
                'Expand to multi-country license plate formats'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/license-plate-detection-yolov9',
            demoLink: 'https://license-plate-detection-demo.streamlit.app',
            tags: ['YOLOv9', 'License Plate Detection', 'ANPR', 'Smart Surveillance', 'Traffic Monitoring'],
            featured: true,
            projectNumber: 40,
            totalProjects: 120,
            categoryProgress: '20/20 DL Projects - COMPLETED! 🎉'
        }
    ],

    // 👁️ Computer Vision Projects
    cv: [
        {
            id: 'cv-1',
            title: 'OCR using OpenCV',
            category: 'Computer Vision',
            domain: 'Document Processing',
            description: 'Implemented Optical Character Recognition system using OpenCV and Tesseract OCR for extracting text from noisy and distorted images. Applied various thresholding techniques and morphological operations for robust text extraction.',
            image: 'port/cv/1.jpg',
            video: 'port/cv/1.mp4',
            technologies: ['Python', 'OpenCV', 'Tesseract OCR', 'NumPy', 'PIL', 'Matplotlib'],
            frameworks: ['OpenCV', 'Tesseract'],
            accuracy: '89%',
            modelSize: '15MB',
            trainingTime: 'No training (Rule-based)',
            dataset: 'Real-world Scanned Text Images',
            keyFeatures: [
                'Multiple thresholding techniques (Otsu, Binary, Adaptive)',
                'Morphological operations for noise removal',
                'Image preprocessing pipeline for OCR optimization',
                'Text extraction from distorted and noisy documents',
                'Real-time text recognition capability'
            ],
            technicalDetails: {
                preprocessing: 'Otsu, Binary, and Adaptive thresholding',
                noiseReduction: 'Morphological operations (erosion, dilation)',
                textExtraction: 'Tesseract OCR engine integration',
                imageEnhancement: 'Contrast adjustment and denoising',
                evaluation: 'Character and word recognition accuracy'
            },
            results: {
                textAccuracy: '89%',
                characterRecognition: '92%',
                wordRecognition: '87%',
                processingSpeed: '1.5 seconds per image'
            },
            applications: [
                'Document digitization systems',
                'Automated form processing',
                'Assistive technology for visually impaired',
                'License plate text extraction'
            ],
            futurePlans: [
                'Integrate deep learning OCR models (EasyOCR, PaddleOCR)',
                'Add multi-language text recognition support',
                'Deploy as web-based OCR service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ocr-opencv-tesseract',
            demoLink: 'https://ocr-text-extraction-demo.streamlit.app',
            tags: ['OCR', 'OpenCV', 'Tesseract', 'Text Extraction', 'Document Processing'],
            featured: true,
            projectNumber: 41,
            totalProjects: 120,
            categoryProgress: '1/20 CV Projects'
        },
        {
            id: 'cv-2',
            title: 'Face Recognition & Detection using Haar Cascade & LBPHFaceRecognizer',
            category: 'Computer Vision',
            domain: 'Biometric Recognition',
            description: 'Implemented face detection and recognition system using Haar Cascade Classifier for detection and LBPHFaceRecognizer for identification. Applied lightweight OpenCV-based approach for real-time face recognition with custom dataset.',
            image: 'port/cv/2.jpg',
            video: 'port/cv/2.mp4',
            technologies: ['Python', 'OpenCV', 'Haar Cascade', 'LBPH', 'NumPy', 'PIL'],
            frameworks: ['OpenCV'],
            accuracy: '87%',
            modelSize: '8MB',
            trainingTime: '30 minutes',
            dataset: 'Custom Face Dataset (Collected & Labeled)',
            keyFeatures: [
                'Haar Cascade Classifier for real-time face detection',
                'LBPH (Local Binary Pattern Histogram) for face recognition',
                'Custom face dataset collection and preprocessing',
                'Real-time video stream face recognition',
                'Lightweight approach suitable for low-resource systems'
            ],
            technicalDetails: {
                detection: 'Haar Cascade frontal face classifier',
                recognition: 'LBPH algorithm for face identification',
                preprocessing: 'Grayscale conversion and histogram equalization',
                training: 'Custom face dataset with data augmentation',
                evaluation: 'Recognition accuracy and detection speed'
            },
            results: {
                recognitionAccuracy: '87%',
                detectionSpeed: '30 FPS',
                falsePositiveRate: '8%',
                processingLatency: '33ms per frame'
            },
            applications: [
                'Security and access control systems',
                'Attendance tracking automation',
                'Personalized user interfaces',
                'Surveillance and monitoring systems'
            ],
            futurePlans: [
                'Upgrade to deep learning face recognition (FaceNet)',
                'Implement multi-face recognition capability',
                'Add face mask detection integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/face-recognition-haar-lbph',
            demoLink: 'https://face-recognition-demo.streamlit.app',
            tags: ['Face Recognition', 'Haar Cascade', 'LBPH', 'Biometric Security', 'Real-time Processing'],
            featured: true,
            projectNumber: 42,
            totalProjects: 120,
            categoryProgress: '2/20 CV Projects'
        },
        {
            id: 'cv-3',
            title: 'Image Colorization using Caffe Model with Deep Learning & OpenCV',
            category: 'Computer Vision',
            domain: 'Image Enhancement',
            description: 'Implemented automatic image colorization using pretrained Caffe deep learning model with OpenCV. Applied LAB color space conversion to transform grayscale images into vibrant colored images with realistic color details.',
            image: 'port/cv/3.jpg',
            video: 'port/cv/3.mp4',
            technologies: ['Python', 'OpenCV', 'Caffe', 'Deep Learning', 'NumPy', 'PIL'],
            frameworks: ['OpenCV', 'Caffe'],
            accuracy: '85%',
            modelSize: '125MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'Large-scale Image Dataset (Pretrained)',
            keyFeatures: [
                'Pretrained Caffe model for automatic colorization',
                'LAB color space conversion for optimal results',
                'Deep learning-based color prediction',
                'Real-time grayscale to color transformation',
                'Historical photo restoration capability'
            ],
            technicalDetails: {
                model: 'Pretrained Caffe deep learning colorization model',
                colorSpace: 'LAB color space (L=Lightness, A/B=Color channels)',
                processing: 'OpenCV DNN module for model inference',
                enhancement: 'Automatic color detail generation',
                evaluation: 'Visual quality and color realism assessment'
            },
            results: {
                colorizationQuality: '85%',
                colorRealism: '82%',
                processingSpeed: '2.1 seconds per image',
                supportedFormats: 'JPG, PNG, BMP'
            },
            applications: [
                'Historical photo restoration',
                'Artistic image enhancement',
                'Film and media colorization',
                'Photo editing and enhancement tools'
            ],
            futurePlans: [
                'Implement custom colorization models',
                'Add user-guided colorization features',
                'Deploy as web-based colorization service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/image-colorization-caffe',
            demoLink: 'https://image-colorization-demo.streamlit.app',
            tags: ['Image Colorization', 'Caffe', 'Deep Learning', 'LAB Color Space', 'Photo Restoration'],
            featured: true,
            projectNumber: 43,
            totalProjects: 120,
            categoryProgress: '3/20 CV Projects'
        },
        {
            id: 'cv-4',
            title: 'Object Detection & Tracking using YOLOv8 & ByteTrack (Supervision)',
            category: 'Computer Vision',
            domain: 'Object Tracking',
            description: 'Implemented multi-object detection and tracking system using YOLOv8 for detection and ByteTrack from Supervision for tracking. Applied Kalman Filter and IoU-based association for robust real-time object tracking with occlusion handling.',
            image: 'port/cv/4.jpg',
            video: 'port/cv/4.mp4',
            technologies: ['Python', 'YOLOv8', 'ByteTrack', 'Supervision', 'OpenCV', 'Ultralytics', 'NumPy'],
            frameworks: ['YOLOv8', 'Supervision', 'OpenCV'],
            accuracy: '91%',
            modelSize: '22MB',
            trainingTime: 'Pretrained (Fine-tuning: 2 hours)',
            dataset: 'COCO, MOT17, Custom Datasets',
            keyFeatures: [
                'YOLOv8 for high-speed object detection',
                'ByteTrack multi-object tracking algorithm',
                'Kalman Filter for motion prediction',
                'IoU-based object association',
                'Real-time tracking with occlusion handling'
            ],
            technicalDetails: {
                detection: 'YOLOv8 for real-time object detection',
                tracking: 'ByteTrack algorithm from Supervision library',
                motionPrediction: 'Kalman Filter for trajectory estimation',
                association: 'IoU-based object matching across frames',
                evaluation: 'MOTA, MOTP, ID switches, tracking accuracy'
            },
            results: {
                detectionAccuracy: '91%',
                trackingAccuracy: '88%',
                processingSpeed: '35 FPS',
                idSwitches: 'Minimal (<2% per sequence)'
            },
            applications: [
                'Smart surveillance and security systems',
                'Autonomous vehicle perception',
                'Sports analytics and player tracking',
                'Traffic monitoring and analysis'
            ],
            futurePlans: [
                'Implement DeepSORT for comparison',
                'Add re-identification features for long-term tracking',
                'Deploy on edge devices for real-time applications'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/object-tracking-yolov8-bytetrack',
            demoLink: 'https://object-tracking-demo.streamlit.app',
            tags: ['YOLOv8', 'ByteTrack', 'Object Tracking', 'Multi-Object Tracking', 'Supervision'],
            featured: true,
            projectNumber: 44,
            totalProjects: 120,
            categoryProgress: '4/20 CV Projects'
        },
        {
            id: 'cv-5',
            title: 'Instance Segmentation for Roads using Detectron2',
            category: 'Computer Vision',
            domain: 'Autonomous Systems',
            description: 'Implemented instance segmentation for road scene understanding using Detectron2 Mask R-CNN. Applied fine-tuning on custom road dataset to segment lanes, potholes, vehicles, pedestrians, and traffic signs with pixel-wise accuracy.',
            image: 'port/cv/5.jpg',
            video: 'port/cv/5.mp4',
            technologies: ['Python', 'Detectron2', 'PyTorch', 'Mask R-CNN', 'OpenCV', 'NumPy', 'COCO API'],
            frameworks: ['Detectron2', 'PyTorch'],
            accuracy: '89%',
            modelSize: '165MB',
            trainingTime: '6 hours',
            dataset: 'Custom Road Dataset (Open Data Sources)',
            keyFeatures: [
                'Mask R-CNN for instance segmentation',
                'Multi-class road object detection and segmentation',
                'Fine-tuning on custom road dataset',
                'Pixel-wise classification of road elements',
                'Real-time road scene understanding'
            ],
            technicalDetails: {
                architecture: 'Mask R-CNN with ResNet-50 backbone',
                segmentation: 'Instance-level pixel-wise classification',
                classes: 'Lanes, Potholes, Vehicles, Pedestrians, Traffic Signs',
                finetuning: 'Transfer learning on custom road dataset',
                evaluation: 'mAP, IoU, Segmentation Accuracy'
            },
            results: {
                segmentationAccuracy: '89%',
                mAP: '0.86',
                iouScore: '0.82',
                processingSpeed: '12 FPS'
            },
            applications: [
                'Autonomous vehicle perception systems',
                'Smart traffic management',
                'Road infrastructure monitoring',
                'Advanced driver assistance systems (ADAS)'
            ],
            futurePlans: [
                'Implement real-time segmentation optimization',
                'Add 3D road scene understanding',
                'Deploy on autonomous vehicle platforms'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/road-segmentation-detectron2',
            demoLink: 'https://road-segmentation-demo.streamlit.app',
            tags: ['Instance Segmentation', 'Detectron2', 'Mask R-CNN', 'Road Segmentation', 'Autonomous Driving'],
            featured: true,
            projectNumber: 45,
            totalProjects: 120,
            categoryProgress: '5/20 CV Projects'
        },
        {
            id: 'cv-6',
            title: 'Aerial Image Classification using ResNet-50',
            category: 'Computer Vision',
            domain: 'Geospatial AI',
            description: 'Implemented aerial image classification using ResNet-50 with transfer learning to classify satellite and aerial imagery into different land types including urban, forest, water bodies, and farmland for geospatial analysis.',
            image: 'port/cv/6.jpg',
            video: 'port/cv/6.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'ResNet-50', 'OpenCV', 'NumPy', 'Matplotlib'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '93%',
            modelSize: '98MB',
            trainingTime: '4 hours',
            dataset: 'Satellite & Aerial Imagery Dataset',
            keyFeatures: [
                'ResNet-50 with ImageNet pretrained weights',
                'Transfer learning for aerial image classification',
                'Multi-class land type classification',
                'Geospatial data preprocessing pipeline',
                'High-resolution satellite image analysis'
            ],
            technicalDetails: {
                architecture: 'ResNet-50 with custom classification head',
                transferLearning: 'ImageNet pretrained weights fine-tuning',
                classes: 'Urban, Forest, Water Bodies, Farmland, Desert, Mountains',
                preprocessing: 'Image normalization and data augmentation',
                evaluation: 'Accuracy, Precision, Recall, F1-Score'
            },
            results: {
                classificationAccuracy: '93%',
                precision: '92%',
                recall: '94%',
                f1Score: '93%'
            },
            applications: [
                'Disaster management and response',
                'Urban planning and development',
                'Agricultural monitoring and crop analysis',
                'Environmental conservation tracking'
            ],
            futurePlans: [
                'Implement semantic segmentation for detailed land analysis',
                'Add change detection capabilities over time',
                'Deploy for real-time satellite image processing'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/aerial-classification-resnet50',
            demoLink: 'https://aerial-classification-demo.streamlit.app',
            tags: ['Aerial Classification', 'ResNet-50', 'Satellite Imagery', 'Geospatial AI', 'Transfer Learning'],
            featured: true,
            projectNumber: 46,
            totalProjects: 120,
            categoryProgress: '6/20 CV Projects'
        },
        {
            id: 'cv-7',
            title: 'AI-Based AR Virtual Makeup',
            category: 'Computer Vision',
            domain: 'Augmented Reality',
            description: 'Developed AI-powered Augmented Reality virtual makeup system using MediaPipe for real-time facial landmark detection and virtual makeup application. Applied computer vision techniques for seamless beauty filter experiences.',
            image: 'port/cv/7.jpg',
            video: 'port/cv/7.mp4',
            technologies: ['Python', 'MediaPipe', 'OpenCV', 'NumPy', 'PIL', 'Face Mesh'],
            frameworks: ['MediaPipe', 'OpenCV'],
            accuracy: '95%',
            modelSize: '12MB',
            trainingTime: 'Pretrained (Real-time inference)',
            dataset: 'Real-time Webcam Input',
            keyFeatures: [
                'Real-time facial landmark detection',
                'Virtual makeup filters (Lipstick, Blush, Eyeliner)',
                'Face tracking and mesh generation',
                'AR-based beauty enhancement',
                'Interactive makeup try-on experience'
            ],
            technicalDetails: {
                faceDetection: 'MediaPipe Face Mesh for 468 facial landmarks',
                tracking: 'Real-time face tracking and alignment',
                filters: 'Virtual makeup application using OpenCV',
                rendering: 'Real-time AR overlay rendering',
                evaluation: 'Tracking accuracy and filter quality'
            },
            results: {
                trackingAccuracy: '95%',
                processingSpeed: '30 FPS',
                landmarkPrecision: '97%',
                filterQuality: '92%'
            },
            applications: [
                'Beauty and cosmetics industry',
                'Virtual try-on e-commerce platforms',
                'Social media AR filters',
                'Beauty consultation and tutorials'
            ],
            futurePlans: [
                'Add 3D makeup effects and lighting',
                'Implement skin tone analysis for personalized recommendations',
                'Deploy as mobile AR application'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ar-virtual-makeup-mediapipe',
            demoLink: 'https://virtual-makeup-demo.streamlit.app',
            tags: ['AR Virtual Makeup', 'MediaPipe', 'Face Tracking', 'Beauty Tech', 'Augmented Reality'],
            featured: true,
            projectNumber: 47,
            totalProjects: 120,
            categoryProgress: '7/20 CV Projects'
        },
        {
            id: 'cv-8',
            title: 'Attendance Marking System using Face Recognition',
            category: 'Computer Vision',
            domain: 'Automation Systems',
            description: 'Developed AI-powered attendance marking system using face recognition integrated with MongoDB for automated attendance tracking. Applied facial embeddings and real-time recognition for fraud-proof attendance management.',
            image: 'port/cv/8.jpg',
            video: 'port/cv/8.mp4',
            technologies: ['Python', 'OpenCV', 'Face Recognition Library', 'MongoDB', 'NumPy', 'Pandas'],
            frameworks: ['OpenCV', 'Face Recognition', 'MongoDB'],
            accuracy: '96%',
            modelSize: '25MB',
            trainingTime: '1 hour',
            dataset: 'Custom Employee/Student Face Database',
            keyFeatures: [
                'Real-time face detection and recognition',
                'Automated attendance logging in MongoDB',
                'Facial embedding-based identification',
                'Anti-spoofing and fraud prevention',
                'Comprehensive attendance reporting system'
            ],
            technicalDetails: {
                faceRecognition: 'Face Recognition library with 128-dimensional embeddings',
                database: 'MongoDB for attendance records and user management',
                detection: 'OpenCV for real-time face detection',
                authentication: 'Facial embedding comparison for identity verification',
                evaluation: 'Recognition accuracy and system reliability'
            },
            results: {
                recognitionAccuracy: '96%',
                processingSpeed: '25 FPS',
                falseAcceptanceRate: '0.1%',
                systemUptime: '99.5%'
            },
            applications: [
                'Educational institution attendance systems',
                'Corporate employee time tracking',
                'Event and conference check-ins',
                'Secure facility access control'
            ],
            futurePlans: [
                'Add mobile app integration for remote attendance',
                'Implement mask-wearing attendance recognition',
                'Deploy cloud-based attendance analytics dashboard'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/attendance-system-face-recognition',
            demoLink: 'https://attendance-system-demo.streamlit.app',
            tags: ['Attendance System', 'Face Recognition', 'MongoDB', 'Automation', 'Biometric Authentication'],
            featured: true,
            projectNumber: 48,
            totalProjects: 120,
            categoryProgress: '8/20 CV Projects'
        },
        {
            id: 'cv-9',
            title: 'Pencil Sketch Conversion',
            category: 'Computer Vision',
            domain: 'Image Processing',
            description: 'Implemented pencil sketch conversion system that transforms any image into hand-drawn sketch using computer vision techniques. Applied grayscale conversion, Gaussian blur, edge detection, and blending for artistic image transformation.',
            image: 'port/cv/9.jpg',
            video: 'port/cv/9.mp4',
            technologies: ['Python', 'OpenCV', 'NumPy', 'PIL', 'Matplotlib'],
            frameworks: ['OpenCV'],
            accuracy: '92%',
            modelSize: '5MB',
            trainingTime: 'No training (Rule-based)',
            dataset: 'Custom Image Collection',
            keyFeatures: [
                'Grayscale conversion for base processing',
                'Gaussian blur for image smoothening',
                'Edge detection for sketch-like effects',
                'Image blending for refined output',
                'Real-time artistic transformation'
            ],
            technicalDetails: {
                preprocessing: 'Grayscale conversion and noise reduction',
                blurring: 'Gaussian blur for smooth transitions',
                edgeDetection: 'Canny edge detection and morphological operations',
                blending: 'Dodge blend mode for pencil sketch effect',
                evaluation: 'Visual quality and artistic similarity assessment'
            },
            results: {
                sketchQuality: '92%',
                processingSpeed: '0.8 seconds per image',
                artisticSimilarity: '89%',
                edgePreservation: '94%'
            },
            applications: [
                'Digital art and creative filters',
                'Social media artistic effects',
                'Photo editing applications',
                'Educational art tools'
            ],
            futurePlans: [
                'Add colored pencil sketch variations',
                'Implement neural style transfer for enhanced results',
                'Deploy as mobile app for real-time sketch conversion'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/pencil-sketch-conversion',
            demoLink: 'https://pencil-sketch-demo.streamlit.app',
            tags: ['Pencil Sketch', 'Image Processing', 'OpenCV', 'Digital Art', 'Artistic Filters'],
            featured: true,
            projectNumber: 49,
            totalProjects: 120,
            categoryProgress: '9/20 CV Projects'
        },
        {
            id: 'cv-10',
            title: 'Object Tracking using OpenCV',
            category: 'Computer Vision',
            domain: 'Video Analytics',
            description: 'Implemented comprehensive object tracking system using multiple OpenCV tracking algorithms including BOOSTING, MIL, KCF, TLD, MedianFlow, CSRT, and MOSSE for real-time moving object tracking in videos.',
            image: 'port/cv/10.jpg',
            video: 'port/cv/10.mp4',
            technologies: ['Python', 'OpenCV', 'NumPy', 'Multiple Tracking Algorithms'],
            frameworks: ['OpenCV'],
            accuracy: '88%',
            modelSize: '10MB',
            trainingTime: 'No training (Algorithm-based)',
            dataset: 'Custom Video Sequences',
            keyFeatures: [
                'Multiple OpenCV tracker implementations',
                'Real-time object tracking with bounding boxes',
                'Frame-by-frame dynamic object following',
                'Tracker performance comparison',
                'Adaptive tracking for various object types'
            ],
            technicalDetails: {
                trackers: 'BOOSTING, MIL, KCF, TLD, MedianFlow, CSRT, MOSSE',
                tracking: 'Single object tracking with bounding box visualization',
                performance: 'Speed vs accuracy comparison across algorithms',
                robustness: 'Handling occlusions and scale changes',
                evaluation: 'Tracking accuracy and computational efficiency'
            },
            results: {
                trackingAccuracy: '88%',
                processingSpeed: '40 FPS (MOSSE), 15 FPS (CSRT)',
                robustness: '85% under occlusions',
                bestPerformer: 'CSRT for accuracy, MOSSE for speed'
            },
            applications: [
                'Surveillance and security systems',
                'Sports analytics and player tracking',
                'Autonomous vehicle object following',
                'Video content analysis'
            ],
            futurePlans: [
                'Implement deep learning-based trackers (DeepSORT)',
                'Add multi-object tracking capabilities',
                'Deploy for real-time surveillance applications'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/object-tracking-opencv',
            demoLink: 'https://object-tracking-opencv-demo.streamlit.app',
            tags: ['Object Tracking', 'OpenCV', 'Video Analytics', 'CSRT', 'KCF'],
            featured: true,
            projectNumber: 50,
            totalProjects: 120,
            categoryProgress: '10/20 CV Projects'
        },
        {
            id: 'cv-11',
            title: 'Moving Object Detection',
            category: 'Computer Vision',
            domain: 'Motion Analysis',
            description: 'Implemented moving object detection system using OpenCV background subtraction methods including MOG2 and KNN algorithms. Applied frame differencing, contour detection, and bounding box visualization for real-time motion analysis.',
            image: 'port/cv/11.jpg',
            video: 'port/cv/11.mp4',
            technologies: ['Python', 'OpenCV', 'MOG2', 'KNN', 'NumPy', 'Background Subtraction'],
            frameworks: ['OpenCV'],
            accuracy: '90%',
            modelSize: '8MB',
            trainingTime: 'No training (Algorithm-based)',
            dataset: 'Custom Video Sequences with Moving Objects',
            keyFeatures: [
                'Background subtraction using MOG2 and KNN',
                'Frame differencing for motion detection',
                'Contour detection and analysis',
                'Real-time bounding box visualization',
                'Adaptive background modeling'
            ],
            technicalDetails: {
                backgroundSubtraction: 'MOG2 (Gaussian Mixture) and KNN algorithms',
                motionDetection: 'Frame differencing and threshold-based detection',
                contourAnalysis: 'Morphological operations and contour filtering',
                visualization: 'Bounding box overlay on detected moving objects',
                evaluation: 'Detection accuracy and false positive rate'
            },
            results: {
                detectionAccuracy: '90%',
                processingSpeed: '35 FPS',
                falsePositiveRate: '5%',
                adaptability: '92% in changing lighting conditions'
            },
            applications: [
                'Surveillance and security monitoring',
                'Traffic flow analysis and monitoring',
                'Wildlife behavior tracking',
                'Anomaly detection in public spaces'
            ],
            futurePlans: [
                'Integrate deep learning for improved accuracy',
                'Add object classification for detected movements',
                'Deploy for smart city surveillance systems'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/moving-object-detection',
            demoLink: 'https://moving-object-detection-demo.streamlit.app',
            tags: ['Moving Object Detection', 'MOG2', 'Background Subtraction', 'Motion Analysis', 'Surveillance'],
            featured: true,
            projectNumber: 51,
            totalProjects: 120,
            categoryProgress: '11/20 CV Projects'
        },
        {
            id: 'cv-12',
            title: 'Virtual Calculator',
            category: 'Computer Vision',
            domain: 'Human-Computer Interaction',
            description: 'Built virtual calculator using OpenCV and MediaPipe for touchless arithmetic operations through hand gesture recognition. Applied real-time finger tracking for number selection and mathematical operations without physical contact.',
            image: 'port/cv/12.jpg',
            video: 'port/cv/12.mp4',
            technologies: ['Python', 'OpenCV', 'MediaPipe', 'Hand Tracking', 'NumPy'],
            frameworks: ['OpenCV', 'MediaPipe'],
            accuracy: '94%',
            modelSize: '15MB',
            trainingTime: 'Pretrained (Real-time inference)',
            dataset: 'Real-time Hand Gesture Input',
            keyFeatures: [
                'Hand gesture recognition using MediaPipe Hands',
                'Real-time finger tracking for input selection',
                'Touchless arithmetic operations (+, -, ×, ÷)',
                'Dynamic UI for calculator display',
                'Gesture-based human-computer interaction'
            ],
            technicalDetails: {
                handTracking: 'MediaPipe Hands for 21 hand landmarks',
                gestureRecognition: 'Finger position analysis for number/operation selection',
                ui: 'OpenCV-based dynamic calculator interface',
                processing: 'Real-time gesture interpretation and calculation',
                evaluation: 'Gesture recognition accuracy and response time'
            },
            results: {
                gestureAccuracy: '94%',
                responseTime: '150ms',
                processingSpeed: '30 FPS',
                operationSuccess: '96%'
            },
            applications: [
                'Contactless computing solutions',
                'Accessibility tools for disabled users',
                'Interactive educational applications',
                'Touchless kiosk interfaces'
            ],
            futurePlans: [
                'Add advanced mathematical functions',
                'Implement voice command integration',
                'Deploy as mobile AR calculator app'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/virtual-calculator-mediapipe',
            demoLink: 'https://virtual-calculator-demo.streamlit.app',
            tags: ['Virtual Calculator', 'Hand Tracking', 'MediaPipe', 'Gesture Recognition', 'Touchless Interface'],
            featured: true,
            projectNumber: 52,
            totalProjects: 120,
            categoryProgress: '12/20 CV Projects'
        },
        {
            id: 'cv-13',
            title: 'AI Graffiti Removal',
            category: 'Computer Vision',
            domain: 'Image Restoration',
            description: 'Developed AI-based graffiti removal system using deep learning and OpenCV for automatic graffiti detection and removal while preserving original structure. Applied CNN-based segmentation and advanced inpainting techniques.',
            image: 'port/cv/13.jpg',
            video: 'port/cv/13.mp4',
            technologies: ['Python', 'OpenCV', 'Deep Learning', 'CNN', 'Image Inpainting', 'NumPy'],
            frameworks: ['OpenCV', 'TensorFlow'],
            accuracy: '87%',
            modelSize: '45MB',
            trainingTime: '6 hours',
            dataset: 'Custom Graffiti Dataset',
            keyFeatures: [
                'CNN-based graffiti detection and segmentation',
                'Advanced inpainting using Telea and Navier-Stokes methods',
                'Background preservation during removal process',
                'Automated image quality enhancement',
                'Real-time graffiti removal capability'
            ],
            technicalDetails: {
                detection: 'Pretrained CNN for graffiti segmentation',
                inpainting: 'OpenCV Telea and Navier-Stokes algorithms',
                restoration: 'Structure-preserving image reconstruction',
                enhancement: 'Post-processing for visual quality improvement',
                evaluation: 'Removal accuracy and background preservation quality'
            },
            results: {
                removalAccuracy: '87%',
                backgroundPreservation: '92%',
                processingTime: '3.2 seconds per image',
                visualQuality: '89%'
            },
            applications: [
                'Smart city maintenance systems',
                'Automated urban cleaning solutions',
                'Historical building restoration',
                'Property management and enhancement'
            ],
            futurePlans: [
                'Implement GANs for better inpainting results',
                'Add real-time video graffiti removal',
                'Deploy as municipal maintenance tool'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ai-graffiti-removal',
            demoLink: 'https://graffiti-removal-demo.streamlit.app',
            tags: ['Graffiti Removal', 'Image Inpainting', 'CNN Segmentation', 'Urban AI', 'Image Restoration'],
            featured: true,
            projectNumber: 53,
            totalProjects: 120,
            categoryProgress: '13/20 CV Projects'
        },
        {
            id: 'cv-14',
            title: 'AI-Powered Drowsiness Detection System',
            category: 'Computer Vision',
            domain: 'Safety Systems',
            description: 'Developed AI-powered drowsiness detection system that monitors eye movements in real-time to detect fatigue and prevent accidents. Applied Eye Aspect Ratio calculation and MediaPipe Face Mesh for comprehensive drowsiness monitoring.',
            image: 'port/cv/14.jpg',
            video: 'port/cv/14.mp4',
            technologies: ['Python', 'OpenCV', 'MediaPipe', 'Face Mesh', 'EAR Algorithm', 'NumPy'],
            frameworks: ['OpenCV', 'MediaPipe'],
            accuracy: '93%',
            modelSize: '18MB',
            trainingTime: 'Pretrained (Real-time inference)',
            dataset: 'Real-time Eye Tracking Data',
            keyFeatures: [
                'Eye Aspect Ratio (EAR) calculation for closure detection',
                'Real-time face and eye tracking using MediaPipe Face Mesh',
                'Alert system with sound and visual warnings',
                'Low-light condition compatibility',
                'Driver fatigue prevention for automotive safety'
            ],
            technicalDetails: {
                eyeTracking: 'MediaPipe Face Mesh for 468 facial landmarks',
                earCalculation: 'Eye Aspect Ratio algorithm for drowsiness detection',
                alertSystem: 'Audio and visual warning mechanisms',
                realTimeProcessing: 'Continuous monitoring with low latency',
                evaluation: 'Detection accuracy and false alarm rate'
            },
            results: {
                detectionAccuracy: '93%',
                responseTime: '200ms',
                falseAlarmRate: '4%',
                processingSpeed: '30 FPS'
            },
            applications: [
                'Automotive driver assistance systems',
                'Workplace safety monitoring',
                'Healthcare fatigue assessment',
                'Transportation safety solutions'
            ],
            futurePlans: [
                'Add yawning detection for enhanced accuracy',
                'Implement machine learning for personalized thresholds',
                'Deploy in commercial vehicle fleets'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/drowsiness-detection-system',
            demoLink: 'https://drowsiness-detection-demo.streamlit.app',
            tags: ['Drowsiness Detection', 'Eye Tracking', 'MediaPipe', 'Safety Systems', 'EAR Algorithm'],
            featured: true,
            projectNumber: 54,
            totalProjects: 120,
            categoryProgress: '14/20 CV Projects'
        },
        {
            id: 'cv-15',
            title: 'Bird Species Detection using YOLOv11',
            category: 'Computer Vision',
            domain: 'Wildlife Conservation',
            description: 'Implemented advanced bird species detection and classification system using YOLOv11 architecture for wildlife conservation and biodiversity monitoring. Applied custom dataset training with edge optimization for real-time deployment.',
            image: 'port/cv/15.jpg',
            video: 'port/cv/15.mp4',
            technologies: ['Python', 'YOLOv11', 'PyTorch', 'OpenCV', 'Ultralytics', 'NumPy'],
            frameworks: ['YOLOv11', 'PyTorch', 'OpenCV'],
            accuracy: '92%',
            modelSize: '25MB',
            trainingTime: '5 hours',
            dataset: 'Custom Bird Species Dataset (50+ species)',
            keyFeatures: [
                'YOLOv11 architecture for bird species detection',
                'Custom dataset training with 50+ bird species',
                'Real-time object detection and species classification',
                'Edge-optimized model for mobile deployment',
                'High-accuracy bounding box localization'
            ],
            technicalDetails: {
                architecture: 'YOLOv11 with custom classification head',
                training: 'Transfer learning from pre-trained YOLOv11 weights',
                optimization: 'Model quantization for edge device deployment',
                augmentation: 'Data augmentation for improved generalization',
                evaluation: 'mAP, Precision, Recall, Species Classification Accuracy'
            },
            results: {
                detectionAccuracy: '92%',
                speciesClassification: '89%',
                mAP: '0.88',
                inferenceSpeed: '45 FPS'
            },
            applications: [
                'Wildlife conservation monitoring',
                'Birdwatching assistance applications',
                'Biodiversity research and studies',
                'Environmental monitoring systems'
            ],
            futurePlans: [
                'Expand dataset to include more rare species',
                'Implement behavior analysis from detection sequences',
                'Deploy as mobile app for field researchers'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/bird-species-detection-yolov11',
            demoLink: 'https://bird-species-detection-demo.streamlit.app',
            tags: ['YOLOv11', 'Bird Detection', 'Species Classification', 'Wildlife Conservation', 'Edge AI'],
            featured: true,
            projectNumber: 55,
            totalProjects: 120,
            categoryProgress: '15/20 CV Projects'
        },
        {
            id: 'cv-16',
            title: 'Barcode Detection and Decoder using Polygon with OpenCV',
            category: 'Computer Vision',
            domain: 'Retail Technology',
            description: 'Developed comprehensive barcode detection and decoding system using OpenCV with polygon approximation for precise localization. Applied contour detection and geometric analysis for robust 1D/QR code scanning in various conditions.',
            image: 'port/cv/16.jpg',
            video: 'port/cv/16.mp4',
            technologies: ['Python', 'OpenCV', 'NumPy', 'Polygon Approximation', 'Contour Detection'],
            frameworks: ['OpenCV'],
            accuracy: '95%',
            modelSize: '12MB',
            trainingTime: 'No training (Algorithm-based)',
            dataset: 'Real-world Barcode Images',
            keyFeatures: [
                'Contour-based barcode detection with polygon approximation',
                'Multi-format support (1D barcodes, QR codes)',
                'Robust performance across angles and lighting conditions',
                'Precise polygon localization around detected codes',
                'Real-time scanning capability for mobile applications'
            ],
            technicalDetails: {
                detection: 'Contour analysis with Douglas-Peucker polygon approximation',
                preprocessing: 'Gaussian blur, thresholding, and morphological operations',
                decoding: 'OpenCV-compatible barcode decoders integration',
                localization: 'Precise polygon boundary detection and visualization',
                evaluation: 'Detection accuracy and decoding success rate'
            },
            results: {
                detectionAccuracy: '95%',
                decodingSuccess: '93%',
                processingSpeed: '40 FPS',
                angleRobustness: '±45 degrees'
            },
            applications: [
                'Retail point-of-sale systems',
                'Inventory management and tracking',
                'Logistics and warehouse automation',
                'Mobile scanning applications'
            ],
            futurePlans: [
                'Add support for DataMatrix and PDF417 codes',
                'Implement batch processing for multiple barcodes',
                'Deploy as mobile SDK for retail applications'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/barcode-detection-polygon-opencv',
            demoLink: 'https://barcode-scanner-demo.streamlit.app',
            tags: ['Barcode Detection', 'OpenCV', 'Polygon Approximation', 'Retail Tech', 'QR Code Scanner'],
            featured: true,
            projectNumber: 56,
            totalProjects: 120,
            categoryProgress: '16/20 CV Projects'
        },
        {
            id: 'cv-17',
            title: 'Video Summarization using OpenAI Whisper',
            category: 'Computer Vision',
            domain: 'Content Analysis',
            description: 'Built intelligent video summarization system using OpenAI Whisper for speech-to-text transcription and transformer-based summarization. Applied FFMPEG for audio extraction and NLP techniques for content condensation.',
            image: 'port/cv/17.jpg',
            video: 'port/cv/17.mp4',
            technologies: ['Python', 'OpenAI Whisper', 'FFMPEG', 'Transformers', 'Hugging Face', 'NLP'],
            frameworks: ['OpenAI Whisper', 'Transformers', 'FFMPEG'],
            accuracy: '91%',
            modelSize: '550MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'Custom Video Content (Meetings, Lectures, YouTube)',
            keyFeatures: [
                'Audio extraction from multiple video formats using FFMPEG',
                'High-accuracy speech-to-text with OpenAI Whisper',
                'Transformer-based text summarization',
                'Multi-language transcription support',
                'Automated key highlight extraction'
            ],
            technicalDetails: {
                audioExtraction: 'FFMPEG for multi-format video to audio conversion',
                transcription: 'OpenAI Whisper for robust speech recognition',
                summarization: 'Transformer-based abstractive summarization',
                preprocessing: 'Audio normalization and noise reduction',
                evaluation: 'Transcription accuracy and summary quality metrics'
            },
            results: {
                transcriptionAccuracy: '91%',
                summarizationQuality: '88%',
                processingSpeed: '2x real-time',
                languageSupport: '99+ languages'
            },
            applications: [
                'Meeting and conference summarization',
                'Educational lecture condensation',
                'YouTube content analysis',
                'Podcast and webinar highlights'
            ],
            futurePlans: [
                'Add speaker diarization for multi-speaker videos',
                'Implement visual content analysis integration',
                'Deploy as web service for content creators'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/video-summarization-whisper',
            demoLink: 'https://video-summarizer-demo.streamlit.app',
            tags: ['Video Summarization', 'OpenAI Whisper', 'Speech-to-Text', 'Content Analysis', 'Transformers'],
            featured: true,
            projectNumber: 57,
            totalProjects: 120,
            categoryProgress: '17/20 CV Projects'
        },
        {
            id: 'cv-18',
            title: 'AI Image Captioning for Social Media Content using Gemini',
            category: 'Computer Vision',
            domain: 'Social Media Automation',
            description: 'Developed AI-powered image caption generator using Google Gemini for creating engaging, platform-specific social media content. Applied multimodal AI for visual context extraction and tone-customized caption generation.',
            image: 'port/cv/18.jpg',
            video: 'port/cv/18.mp4',
            technologies: ['Python', 'Google Gemini API', 'Pillow', 'OpenCV', 'Transformers', 'Multimodal AI'],
            frameworks: ['Google Gemini', 'OpenCV', 'Pillow'],
            accuracy: '89%',
            modelSize: '2GB (Cloud-based)',
            trainingTime: 'Pretrained (API-based)',
            dataset: 'Social Media Images (Instagram, LinkedIn, WhatsApp)',
            keyFeatures: [
                'Google Gemini multimodal AI for visual understanding',
                'Platform-specific caption generation (Instagram, LinkedIn, WhatsApp)',
                'Tone customization (professional, casual, creative)',
                'Automated hashtag generation and optimization',
                'Batch processing for multiple images'
            ],
            technicalDetails: {
                visualAnalysis: 'Google Gemini vision model for image understanding',
                captionGeneration: 'Context-aware text generation with tone adaptation',
                platformOptimization: 'Platform-specific formatting and hashtag strategies',
                preprocessing: 'Image normalization and format conversion',
                evaluation: 'Caption relevance and engagement prediction'
            },
            results: {
                captionRelevance: '89%',
                engagementPrediction: '85%',
                processingSpeed: '3 seconds per image',
                platformSupport: '3 major platforms'
            },
            applications: [
                'Social media content automation',
                'Influencer marketing tools',
                'Brand content generation',
                'E-commerce product descriptions'
            ],
            futurePlans: [
                'Add video content captioning support',
                'Implement A/B testing for caption optimization',
                'Deploy as browser extension for content creators'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ai-image-captioning-gemini',
            demoLink: 'https://social-media-captioner-demo.streamlit.app',
            tags: ['Image Captioning', 'Google Gemini', 'Social Media Automation', 'Multimodal AI', 'Content Creation'],
            featured: true,
            projectNumber: 58,
            totalProjects: 120,
            categoryProgress: '18/20 CV Projects'
        },
        {
            id: 'cv-19',
            title: 'Speed Estimation using Computer Vision',
            category: 'Computer Vision',
            domain: 'Smart Transportation',
            description: 'Developed AI-based vehicle speed estimation system using computer vision techniques for traffic monitoring and surveillance. Applied object tracking with pixel-to-meter conversion for accurate real-time speed calculation.',
            image: 'port/cv/19.jpg',
            video: 'port/cv/19.mp4',
            technologies: ['Python', 'OpenCV', 'NumPy', 'Object Tracking', 'Distance-Time Formula'],
            frameworks: ['OpenCV'],
            accuracy: '91%',
            modelSize: '15MB',
            trainingTime: 'No training (Algorithm-based)',
            dataset: 'Traffic Video Surveillance Data',
            keyFeatures: [
                'Real-time vehicle detection and tracking',
                'Pixel-to-meter conversion for distance calculation',
                'Speed estimation using distance-time formula',
                'CSV export for speed logging and analysis',
                'Multi-vehicle simultaneous speed monitoring'
            ],
            technicalDetails: {
                detection: 'OpenCV-based vehicle detection algorithms',
                tracking: 'Kalman Filter and centroid tracking',
                calibration: 'Camera calibration for pixel-to-meter conversion',
                speedCalculation: 'Distance/Time formula with frame rate consideration',
                evaluation: 'Speed accuracy comparison with ground truth'
            },
            results: {
                speedAccuracy: '91%',
                processingSpeed: '25 FPS',
                detectionRange: '5-150 km/h',
                multiVehicleSupport: 'Up to 15 vehicles simultaneously'
            },
            applications: [
                'Traffic monitoring and enforcement',
                'Highway surveillance systems',
                'Smart city traffic management',
                'Automated speed violation detection'
            ],
            futurePlans: [
                'Integrate with YOLO for improved vehicle detection',
                'Add license plate recognition for violation tracking',
                'Deploy on edge devices for real-time monitoring'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/speed-estimation-computer-vision',
            demoLink: 'https://speed-estimator-demo.streamlit.app',
            tags: ['Speed Estimation', 'Traffic Monitoring', 'Object Tracking', 'Smart Transportation', 'Surveillance AI'],
            featured: true,
            projectNumber: 59,
            totalProjects: 120,
            categoryProgress: '19/20 CV Projects'
        },
        {
            id: 'cv-20',
            title: 'Self-Supervised Learning using SimCLR',
            category: 'Computer Vision',
            domain: 'Representation Learning',
            description: 'Implemented SimCLR (Simple Contrastive Learning of Representations) framework for self-supervised learning without labeled data. Applied contrastive loss and data augmentation to learn high-quality visual representations.',
            image: 'port/cv/20.jpg',
            video: 'port/cv/20.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'SimCLR', 'Contrastive Learning', 'Data Augmentation'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '87%',
            modelSize: '95MB',
            trainingTime: '8 hours',
            dataset: 'CIFAR-10, STL-10 (Unlabeled)',
            keyFeatures: [
                'SimCLR contrastive learning framework implementation',
                'Self-supervised representation learning without labels',
                'Advanced data augmentation pipeline',
                'Linear evaluation protocol for representation quality',
                'Transfer learning capabilities for downstream tasks'
            ],
            technicalDetails: {
                architecture: 'ResNet encoder with projection head',
                contrastiveLoss: 'NT-Xent (Normalized Temperature-scaled Cross Entropy)',
                augmentation: 'Random crop, color jitter, Gaussian blur, flip',
                evaluation: 'Linear classifier on frozen representations',
                optimization: 'Large batch training with momentum optimizer'
            },
            results: {
                representationQuality: '87%',
                linearEvaluation: '85% on CIFAR-10',
                transferLearning: '82% on downstream tasks',
                convergenceTime: '8 hours on GPU'
            },
            applications: [
                'Unsupervised feature learning',
                'Transfer learning for limited labeled data',
                'Pre-training for computer vision tasks',
                'Representation learning research'
            ],
            futurePlans: [
                'Implement SimCLRv2 and SwAV variants',
                'Explore vision transformers with contrastive learning',
                'Apply to medical imaging with limited labels'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/simclr-self-supervised-learning',
            demoLink: 'https://simclr-representation-demo.streamlit.app',
            tags: ['SimCLR', 'Self-Supervised Learning', 'Contrastive Learning', 'Representation Learning', 'Transfer Learning'],
            featured: true,
            projectNumber: 60,
            totalProjects: 120,
            categoryProgress: '20/20 CV Projects - COMPLETED! 🎉'
        }
    ],

    // 💬 Natural Language Processing Projects
    nlp: [
        {
            id: 'nlp-1',
            title: 'Text Classification using Naive Bayes (MultinomialNB)',
            category: 'Natural Language Processing',
            domain: 'Text Classification',
            description: 'Implemented text classification system using Multinomial Naive Bayes algorithm for efficient document categorization. Applied TF-IDF vectorization and comprehensive preprocessing for robust text-based predictions.',
            image: 'port/nlp/1.jpg',
            video: 'port/nlp/1.mp4',
            technologies: ['Python', 'Scikit-learn', 'MultinomialNB', 'TF-IDF', 'NLTK', 'Pandas'],
            frameworks: ['Scikit-learn', 'NLTK'],
            accuracy: '88%',
            modelSize: '5MB',
            trainingTime: '15 minutes',
            dataset: 'Labeled Text Classification Dataset',
            keyFeatures: [
                'Multinomial Naive Bayes classification algorithm',
                'TF-IDF vectorization for feature extraction',
                'Comprehensive text preprocessing pipeline',
                'Stopword removal and text normalization',
                'Support for binary and multi-class classification'
            ],
            technicalDetails: {
                algorithm: 'Multinomial Naive Bayes with Laplace smoothing',
                preprocessing: 'Tokenization, stopword removal, lowercasing',
                vectorization: 'TF-IDF (Term Frequency-Inverse Document Frequency)',
                evaluation: 'Accuracy, Precision, Recall, F1-Score',
                optimization: 'Hyperparameter tuning for alpha smoothing'
            },
            results: {
                accuracy: '88%',
                precision: '87%',
                recall: '89%',
                f1Score: '88%'
            },
            applications: [
                'Email spam detection systems',
                'News article categorization',
                'Sentiment analysis for reviews',
                'Document classification automation'
            ],
            futurePlans: [
                'Compare with SVM and Random Forest classifiers',
                'Implement n-gram features for better context',
                'Deploy as real-time text classification API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/text-classification-naive-bayes',
            demoLink: 'https://text-classifier-nb-demo.streamlit.app',
            tags: ['Naive Bayes', 'Text Classification', 'TF-IDF', 'MultinomialNB', 'Document Categorization'],
            featured: true,
            projectNumber: 61,
            totalProjects: 120,
            categoryProgress: '1/20 NLP Projects'
        },
        {
            id: 'nlp-2',
            title: 'Sentiment Analysis using BERT',
            category: 'Natural Language Processing',
            domain: 'Sentiment Analysis',
            description: 'Fine-tuned BERT transformer model for advanced sentiment analysis using Hugging Face Transformers. Applied transfer learning with Trainer API for high-accuracy emotion detection in text data.',
            image: 'port/nlp/2.jpg',
            video: 'port/nlp/2.mp4',
            technologies: ['Python', 'BERT', 'Transformers', 'Hugging Face', 'PyTorch', 'Trainer API'],
            frameworks: ['Hugging Face Transformers', 'PyTorch'],
            accuracy: '94%',
            modelSize: '440MB',
            trainingTime: '3 hours',
            dataset: 'Labeled Sentiment Dataset (Positive/Negative/Neutral)',
            keyFeatures: [
                'BERT fine-tuning for sentiment classification',
                'Hugging Face Trainer API for efficient training',
                'Advanced tokenization with attention masks',
                'Transfer learning from pretrained BERT weights',
                'Multi-class sentiment detection (positive/negative/neutral)'
            ],
            technicalDetails: {
                architecture: 'BERT-base-uncased with classification head',
                finetuning: 'Transfer learning with frozen/unfrozen layers',
                tokenization: 'BERT WordPiece tokenizer with attention masks',
                training: 'Hugging Face Trainer API with custom metrics',
                evaluation: 'Accuracy, F1-Score, Confusion Matrix'
            },
            results: {
                accuracy: '94%',
                f1Score: '93%',
                precision: '94%',
                recall: '93%'
            },
            applications: [
                'Social media sentiment monitoring',
                'Customer feedback analysis',
                'Product review classification',
                'Brand reputation management'
            ],
            futurePlans: [
                'Compare with RoBERTa and DistilBERT models',
                'Implement aspect-based sentiment analysis',
                'Deploy as real-time sentiment analysis API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/sentiment-analysis-bert',
            demoLink: 'https://bert-sentiment-demo.streamlit.app',
            tags: ['BERT', 'Sentiment Analysis', 'Transformers', 'Fine-tuning', 'Transfer Learning'],
            featured: true,
            projectNumber: 62,
            totalProjects: 120,
            categoryProgress: '2/20 NLP Projects'
        },
        {
            id: 'nlp-3',
            title: 'Named Entity Recognition (NER) using SpaCy',
            category: 'Natural Language Processing',
            domain: 'Information Extraction',
            description: 'Built comprehensive Named Entity Recognition system using SpaCy framework for automatic entity detection and labeling. Applied pretrained models with custom pipelines and interactive visualization using displaCy.',
            image: 'port/nlp/3.jpg',
            video: 'port/nlp/3.mp4',
            technologies: ['Python', 'SpaCy', 'displaCy', 'NER', 'Entity Extraction', 'Text Mining'],
            frameworks: ['SpaCy'],
            accuracy: '91%',
            modelSize: '50MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'Custom Text Documents and News Articles',
            keyFeatures: [
                'Automatic entity detection (PERSON, ORG, GPE, MONEY, etc.)',
                'SpaCy pretrained models with custom pipeline configuration',
                'Interactive entity visualization using displaCy',
                'Multi-language entity recognition support',
                'Custom entity type training and fine-tuning'
            ],
            technicalDetails: {
                model: 'SpaCy en_core_web_sm pretrained model',
                entities: 'PERSON, ORG, GPE, MONEY, DATE, TIME, PERCENT',
                pipeline: 'Custom NLP pipeline with tokenizer, tagger, parser, NER',
                visualization: 'displaCy for interactive entity highlighting',
                evaluation: 'Precision, Recall, F1-Score per entity type'
            },
            results: {
                accuracy: '91%',
                personRecognition: '94%',
                organizationRecognition: '89%',
                locationRecognition: '92%'
            },
            applications: [
                'Resume screening and candidate analysis',
                'News article information extraction',
                'Document automation and processing',
                'Knowledge graph construction'
            ],
            futurePlans: [
                'Train custom entity types for domain-specific use cases',
                'Implement relation extraction between entities',
                'Deploy as real-time document processing API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ner-spacy-entity-recognition',
            demoLink: 'https://ner-spacy-demo.streamlit.app',
            tags: ['NER', 'SpaCy', 'Entity Recognition', 'Information Extraction', 'Text Mining'],
            featured: true,
            projectNumber: 63,
            totalProjects: 120,
            categoryProgress: '3/20 NLP Projects'
        },
        {
            id: 'nlp-4',
            title: 'Text Summarization using T5 Base Model + Gradio App',
            category: 'Natural Language Processing',
            domain: 'Text Summarization',
            description: 'Built intelligent text summarization system using Google T5-base transformer model with interactive Gradio interface. Applied sequence-to-sequence learning for generating concise summaries from long-form content.',
            image: 'port/nlp/4.jpg',
            video: 'port/nlp/4.mp4',
            technologies: ['Python', 'T5-base', 'Transformers', 'Gradio', 'Hugging Face', 'PyTorch'],
            frameworks: ['Hugging Face Transformers', 'Gradio'],
            accuracy: '89%',
            modelSize: '850MB',
            trainingTime: 'Pretrained + Fine-tuning (2 hours)',
            dataset: 'CNN/DailyMail, Custom Long-form Text',
            keyFeatures: [
                'T5-base transformer for abstractive summarization',
                'Interactive Gradio web interface for real-time testing',
                'Customizable summary length and parameters',
                'Support for various document types (blogs, news, research)',
                'Fine-tuning capability for domain-specific summarization'
            ],
            technicalDetails: {
                architecture: 'T5-base encoder-decoder transformer',
                tokenization: 'SentencePiece tokenizer with T5 preprocessing',
                generation: 'Beam search and nucleus sampling for summary generation',
                interface: 'Gradio web app with customizable parameters',
                evaluation: 'ROUGE scores and human evaluation metrics'
            },
            results: {
                rougeL: '0.42',
                rouge1: '0.48',
                rouge2: '0.35',
                summaryQuality: '89%'
            },
            applications: [
                'News article summarization',
                'Research paper abstract generation',
                'Legal document condensation',
                'Blog and content summarization'
            ],
            futurePlans: [
                'Implement BART and Pegasus models for comparison',
                'Add multi-document summarization capability',
                'Deploy as production-ready API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/text-summarization-t5-gradio',
            demoLink: 'https://t5-summarizer-gradio-demo.huggingface.co',
            tags: ['T5 Model', 'Text Summarization', 'Gradio', 'Transformers', 'Abstractive Summarization'],
            featured: true,
            projectNumber: 64,
            totalProjects: 120,
            categoryProgress: '4/20 NLP Projects'
        },
        {
            id: 'nlp-5',
            title: 'Question Answering using BERT + Gradio App',
            category: 'Natural Language Processing',
            domain: 'Question Answering',
            description: 'Developed intelligent Question Answering system using BERT transformer with Hugging Face pipeline and interactive Gradio interface. Applied reading comprehension for accurate answer extraction from context.',
            image: 'port/nlp/5.jpg',
            video: 'port/nlp/5.mp4',
            technologies: ['Python', 'BERT', 'Transformers', 'Gradio', 'Hugging Face', 'Pipeline API'],
            frameworks: ['Hugging Face Transformers', 'Gradio'],
            accuracy: '92%',
            modelSize: '440MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'SQuAD 2.0, Custom Context-Question Pairs',
            keyFeatures: [
                'BERT-based extractive question answering',
                'Hugging Face pipeline for streamlined inference',
                'Interactive Gradio web interface for real-time Q&A',
                'Context-aware answer extraction with confidence scores',
                'Support for multiple question types and domains'
            ],
            technicalDetails: {
                architecture: 'BERT-large-uncased-whole-word-masking-finetuned-squad',
                pipeline: 'Hugging Face question-answering pipeline',
                processing: 'Context tokenization and span prediction',
                interface: 'Gradio with context input and question fields',
                evaluation: 'Exact Match (EM) and F1 score metrics'
            },
            results: {
                exactMatch: '85%',
                f1Score: '92%',
                confidenceAccuracy: '89%',
                responseTime: '0.8 seconds'
            },
            applications: [
                'Educational Q&A systems',
                'Customer support automation',
                'Knowledge base querying',
                'Document-based information retrieval'
            ],
            futurePlans: [
                'Implement multi-document question answering',
                'Add conversational context for follow-up questions',
                'Deploy as enterprise knowledge management system'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/question-answering-bert-gradio',
            demoLink: 'https://bert-qa-gradio-demo.huggingface.co',
            tags: ['Question Answering', 'BERT', 'Gradio', 'Reading Comprehension', 'Information Extraction'],
            featured: true,
            projectNumber: 65,
            totalProjects: 120,
            categoryProgress: '5/20 NLP Projects'
        },
        {
            id: 'nlp-6',
            title: 'Topic Modeling using LDA (Latent Dirichlet Allocation)',
            category: 'Natural Language Processing',
            domain: 'Topic Modeling',
            description: 'Built comprehensive topic modeling system using Latent Dirichlet Allocation to discover hidden topics in document collections. Applied Gensim and NLTK with interactive pyLDAvis visualization for topic interpretability.',
            image: 'port/nlp/6.jpg',
            video: 'port/nlp/6.mp4',
            technologies: ['Python', 'Gensim', 'LDA', 'NLTK', 'pyLDAvis', 'Pandas'],
            frameworks: ['Gensim', 'NLTK'],
            accuracy: '87%',
            modelSize: '25MB',
            trainingTime: '45 minutes',
            dataset: 'News Articles, Research Papers, Custom Document Collection',
            keyFeatures: [
                'Latent Dirichlet Allocation for unsupervised topic discovery',
                'Comprehensive text preprocessing with NLTK',
                'Interactive topic visualization using pyLDAvis',
                'Optimal topic number selection using coherence scores',
                'Document-topic and topic-word probability distributions'
            ],
            technicalDetails: {
                algorithm: 'LDA with Dirichlet priors for topic and word distributions',
                preprocessing: 'Tokenization, stopword removal, lemmatization, bigrams',
                modeling: 'Gensim LdaModel with multicore processing',
                evaluation: 'Coherence score, perplexity, and topic interpretability',
                visualization: 'pyLDAvis for interactive topic exploration'
            },
            results: {
                coherenceScore: '0.52',
                topicCoherence: '87%',
                optimalTopics: '8-12 topics',
                interpretability: '85%'
            },
            applications: [
                'Content categorization and organization',
                'Document clustering and similarity analysis',
                'News article topic discovery',
                'Research paper theme identification'
            ],
            futurePlans: [
                'Implement BERTopic for transformer-based topic modeling',
                'Add dynamic topic modeling for temporal analysis',
                'Deploy as document analysis web service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/topic-modeling-lda-gensim',
            demoLink: 'https://lda-topic-modeling-demo.streamlit.app',
            tags: ['LDA', 'Topic Modeling', 'Gensim', 'Unsupervised Learning', 'Text Mining'],
            featured: true,
            projectNumber: 66,
            totalProjects: 120,
            categoryProgress: '6/20 NLP Projects'
        },
        {
            id: 'nlp-7',
            title: 'Multilingual Sentiment Analysis with Gradio App',
            category: 'Natural Language Processing',
            domain: 'Cross-lingual NLP',
            description: 'Developed comprehensive multilingual sentiment analysis system using XLM-RoBERTa and mBERT transformers with interactive Gradio interface. Applied automatic language detection and real-time sentiment prediction across multiple languages.',
            image: 'port/nlp/7.jpg',
            video: 'port/nlp/7.mp4',
            technologies: ['Python', 'XLM-RoBERTa', 'mBERT', 'Transformers', 'Gradio', 'langdetect', 'Hugging Face'],
            frameworks: ['Hugging Face Transformers', 'Gradio', 'langdetect'],
            accuracy: '90%',
            modelSize: '560MB',
            trainingTime: '4 hours',
            dataset: 'Multilingual Sentiment Dataset (English, Hindi, French, Spanish)',
            keyFeatures: [
                'XLM-RoBERTa and mBERT for cross-lingual sentiment understanding',
                'Interactive Gradio web app for real-time multilingual testing',
                'Automatic language detection and preprocessing pipeline',
                'Support for 100+ languages with unified model architecture',
                'Translation integration for unsupported languages'
            ],
            technicalDetails: {
                architecture: 'XLM-RoBERTa-base with multilingual classification head',
                languageDetection: 'langdetect library for automatic language identification',
                preprocessing: 'Language-specific tokenization and normalization',
                training: 'Cross-lingual fine-tuning on multilingual datasets',
                evaluation: 'Per-language accuracy and cross-lingual generalization'
            },
            results: {
                overallAccuracy: '90%',
                englishAccuracy: '93%',
                hindiAccuracy: '88%',
                frenchAccuracy: '89%',
                spanishAccuracy: '91%'
            },
            applications: [
                'Global customer feedback analysis',
                'Cross-lingual social media monitoring',
                'International brand sentiment tracking',
                'Multilingual review classification'
            ],
            futurePlans: [
                'Add support for low-resource languages',
                'Implement zero-shot cross-lingual transfer',
                'Deploy as multilingual sentiment API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/multilingual-sentiment-analysis',
            demoLink: 'https://multilingual-sentiment-demo.streamlit.app',
            tags: ['Multilingual NLP', 'XLM-RoBERTa', 'Cross-lingual', 'Sentiment Analysis', 'Global AI'],
            featured: true,
            projectNumber: 67,
            totalProjects: 120,
            categoryProgress: '7/20 NLP Projects'
        },
        {
            id: 'nlp-8',
            title: 'Text Generation using GPT-Neo 7B',
            category: 'Natural Language Processing',
            domain: 'Text Generation',
            description: 'Built advanced text generation system using GPT-Neo 7B from EleutherAI for creative writing and content creation. Applied zero-shot and few-shot learning with GPU-optimized inference for human-like text generation.',
            image: 'port/nlp/8.jpg',
            video: 'port/nlp/8.mp4',
            technologies: ['Python', 'GPT-Neo 7B', 'EleutherAI', 'Transformers', 'Hugging Face', 'GPU Acceleration'],
            frameworks: ['Hugging Face Transformers', 'EleutherAI'],
            accuracy: '91%',
            modelSize: '14GB',
            trainingTime: 'Pretrained (Fine-tuning: 6 hours)',
            dataset: 'The Pile Dataset, Custom Creative Writing Corpus',
            keyFeatures: [
                'GPT-Neo 7B large language model for text generation',
                'Zero-shot and few-shot learning capabilities',
                'Context-aware paragraph continuation with human-like fluency',
                'GPU-accelerated inference optimization',
                'Fine-tuning for creative writing and storytelling'
            ],
            technicalDetails: {
                architecture: 'GPT-Neo 7B transformer with 7 billion parameters',
                generation: 'Autoregressive text generation with nucleus sampling',
                optimization: 'GPU acceleration with mixed precision training',
                finetuning: 'Domain-specific fine-tuning on creative writing datasets',
                evaluation: 'Perplexity, BLEU score, and human evaluation metrics'
            },
            results: {
                textQuality: '91%',
                perplexity: '18.5',
                coherenceScore: '89%',
                creativityRating: '92%'
            },
            applications: [
                'Creative writing and storytelling assistance',
                'Blog and article content generation',
                'Marketing copy and advertising text creation',
                'Educational content and tutorial writing'
            ],
            futurePlans: [
                'Implement GPT-NeoX and larger models',
                'Add fine-tuning for specific domains (legal, medical)',
                'Deploy as content creation API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/text-generation-gpt-neo-7b',
            demoLink: 'https://gpt-neo-text-generator-demo.streamlit.app',
            tags: ['GPT-Neo', 'Text Generation', 'LLM', 'EleutherAI', 'Creative Writing'],
            featured: true,
            projectNumber: 68,
            totalProjects: 120,
            categoryProgress: '8/20 NLP Projects'
        },
        {
            id: 'nlp-9',
            title: 'Text Clustering using KMeans',
            category: 'Natural Language Processing',
            domain: 'Text Clustering',
            description: 'Implemented unsupervised text clustering system using KMeans algorithm for grouping unlabeled text data by topic similarity. Applied TF-IDF vectorization and Elbow Method for optimal cluster determination.',
            image: 'port/nlp/9.jpg',
            video: 'port/nlp/9.mp4',
            technologies: ['Python', 'KMeans', 'Scikit-learn', 'TF-IDF', 'Elbow Method', 'Pandas'],
            frameworks: ['Scikit-learn'],
            accuracy: '84%',
            modelSize: '8MB',
            trainingTime: '25 minutes',
            dataset: 'News Articles, Customer Feedback, Support Tickets',
            keyFeatures: [
                'KMeans clustering for unsupervised text grouping',
                'TF-IDF vectorization for feature extraction',
                'Elbow Method for optimal cluster number selection',
                'Topic similarity analysis and visualization',
                'Large-scale text corpus organization'
            ],
            technicalDetails: {
                algorithm: 'KMeans clustering with Euclidean distance',
                vectorization: 'TF-IDF (Term Frequency-Inverse Document Frequency)',
                optimization: 'Elbow Method and Silhouette Score for cluster validation',
                preprocessing: 'Text normalization, stopword removal, stemming',
                evaluation: 'Silhouette Score, Inertia, and cluster coherence'
            },
            results: {
                clusteringAccuracy: '84%',
                silhouetteScore: '0.68',
                optimalClusters: '6-8 clusters',
                topicCoherence: '82%'
            },
            applications: [
                'News article topic organization',
                'Customer feedback categorization',
                'Support ticket routing and classification',
                'Document management and retrieval systems'
            ],
            futurePlans: [
                'Implement DBSCAN and Hierarchical clustering',
                'Add semantic clustering with sentence embeddings',
                'Deploy as document organization API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/text-clustering-kmeans',
            demoLink: 'https://text-clustering-demo.streamlit.app',
            tags: ['Text Clustering', 'KMeans', 'TF-IDF', 'Unsupervised Learning', 'Topic Similarity'],
            featured: true,
            projectNumber: 69,
            totalProjects: 120,
            categoryProgress: '9/20 NLP Projects'
        },
        {
            id: 'nlp-10',
            title: 'Speech to Text Conversion using OpenAI Whisper',
            category: 'Natural Language Processing',
            domain: 'Speech Recognition',
            description: 'Implemented high-accuracy speech-to-text conversion system using OpenAI Whisper ASR model. Applied multi-format audio processing with automatic language detection for robust voice transcription.',
            image: 'port/nlp/10.jpg',
            video: 'port/nlp/10.mp4',
            technologies: ['Python', 'OpenAI Whisper', 'ASR', 'Audio Processing', 'Language Detection'],
            frameworks: ['OpenAI Whisper'],
            accuracy: '96%',
            modelSize: '550MB',
            trainingTime: 'Pretrained (Inference only)',
            dataset: 'Multi-language Audio Files (English, Spanish, French, etc.)',
            keyFeatures: [
                'OpenAI Whisper for high-accuracy speech recognition',
                'Multi-format audio processing (WAV, MP3, M4A, etc.)',
                'Automatic language detection and transcription',
                'Real-time and batch audio processing capabilities',
                'Robust performance across different accents and noise levels'
            ],
            technicalDetails: {
                model: 'OpenAI Whisper (base/small/medium/large variants)',
                preprocessing: 'Audio normalization and format conversion',
                languageDetection: 'Automatic language identification',
                transcription: 'Transformer-based sequence-to-sequence ASR',
                evaluation: 'Word Error Rate (WER) and transcription accuracy'
            },
            results: {
                transcriptionAccuracy: '96%',
                wordErrorRate: '4%',
                languageSupport: '99+ languages',
                processingSpeed: '2x real-time'
            },
            applications: [
                'Voice assistants and smart speakers',
                'Audio documentation and meeting transcription',
                'Accessibility tools for hearing impaired',
                'Podcast and video subtitle generation'
            ],
            futurePlans: [
                'Implement real-time streaming transcription',
                'Add speaker diarization capabilities',
                'Deploy as voice processing API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/speech-to-text-whisper',
            demoLink: 'https://whisper-speech-to-text-demo.streamlit.app',
            tags: ['Speech to Text', 'OpenAI Whisper', 'ASR', 'Voice Recognition', 'Audio Processing'],
            featured: true,
            projectNumber: 70,
            totalProjects: 120,
            categoryProgress: '10/20 NLP Projects'
        },
        {
            id: 'nlp-11',
            title: 'Question Generation using LLaMA-8B (Meta AI)',
            category: 'Natural Language Processing',
            domain: 'Question Generation',
            description: 'Built intelligent question generation system using LLaMA-8B from Meta AI for automated educational content creation. Applied fine-tuning with Streamlit interface for real-time text-to-question conversion.',
            image: 'port/nlp/11.jpg',
            video: 'port/nlp/11.mp4',
            technologies: ['Python', 'LLaMA-8B', 'Meta AI', 'Streamlit', 'Transformers', 'Fine-tuning'],
            frameworks: ['LLaMA', 'Streamlit', 'Hugging Face'],
            accuracy: '89%',
            modelSize: '16GB',
            trainingTime: 'Pretrained + Fine-tuning (8 hours)',
            dataset: 'Educational Text Passages, Q&A Datasets',
            keyFeatures: [
                'LLaMA-8B large language model for question generation',
                'Context-aware intelligent question formulation',
                'Fine-tuning on educational and assessment datasets',
                'Interactive Streamlit web application',
                'Real-time text-to-question conversion'
            ],
            technicalDetails: {
                architecture: 'LLaMA-8B transformer with 8 billion parameters',
                finetuning: 'Domain-specific fine-tuning on question generation tasks',
                interface: 'Streamlit web app with passage input and question output',
                generation: 'Autoregressive question generation with beam search',
                evaluation: 'Question quality, relevance, and educational value metrics'
            },
            results: {
                questionQuality: '89%',
                contextRelevance: '92%',
                educationalValue: '87%',
                generationSpeed: '3 seconds per question'
            },
            applications: [
                'Educational technology platforms',
                'Automated exam and quiz generation',
                'Study material and assessment creation',
                'Interactive learning and tutoring systems'
            ],
            futurePlans: [
                'Implement multi-level question difficulty generation',
                'Add support for different question types (MCQ, short answer)',
                'Deploy as educational API for learning platforms'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/question-generation-llama-8b',
            demoLink: 'https://llama-question-generator-demo.streamlit.app',
            tags: ['LLaMA', 'Question Generation', 'EdTech', 'Streamlit', 'Educational AI'],
            featured: true,
            projectNumber: 71,
            totalProjects: 120,
            categoryProgress: '11/20 NLP Projects'
        },
        {
            id: 'nlp-12',
            title: 'Fake News Detection using Agentic AI',
            category: 'Natural Language Processing',
            domain: 'Agentic AI & Misinformation Detection',
            description: 'Developed a robust Fake News Detection System using the power of Agentic AI, combining multiple modalities and intelligent agents for comprehensive misinformation analysis.',
            image: 'port/nlp/12.jpg',
            video: 'port/nlp/12.mp4',
            technologies: ['Python', 'Transformers', 'Tesseract OCR', 'Speech-to-Text', 'BERT', 'LangChain', 'OpenAI API', 'Scikit-learn', 'OpenCV', 'Streamlit'],
            frameworks: ['LangChain', 'Transformers', 'OpenCV', 'Streamlit'],
            accuracy: '94%',
            modelSize: '680MB',
            trainingTime: '6 hours',
            dataset: 'Custom Multi-modal Fake News Dataset (50K+ articles)',
            keyFeatures: [
                'Agentic AI architecture with intelligent agent coordination',
                'Real-time news generation via LLMs for testing',
                'Live fake news detection using NLP-based classifiers',
                'Audio input support with real-time speech-to-text conversion',
                'Image-based text detection using Tesseract OCR for scanned news',
                'Multi-modal input processing (text, audio, image)',
                'Automated fact-checking with external source verification'
            ],
            technicalDetails: {
                agenticArchitecture: 'Multi-agent system with specialized detection agents',
                textAnalysis: 'BERT-based transformer for semantic analysis',
                audioProcessing: 'Whisper ASR for speech-to-text conversion',
                imageProcessing: 'Tesseract OCR for text extraction from images',
                factChecking: 'LangChain integration with external knowledge bases',
                evaluation: 'Multi-modal accuracy assessment and confidence scoring'
            },
            results: {
                overallAccuracy: '94%',
                textDetection: '96%',
                audioDetection: '91%',
                imageDetection: '89%',
                factCheckingAccuracy: '92%'
            },
            applications: [
                'Social media misinformation detection',
                'News verification and fact-checking platforms',
                'Automated content moderation systems',
                'Educational media literacy tools'
            ],
            futurePlans: [
                'Implement video-based fake news detection',
                'Add multilingual support for global deployment',
                'Deploy as browser extension for real-time verification'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/fake-news-agentic-ai',
            demoLink: 'https://fake-news-agentic-ai-demo.streamlit.app',
            tags: ['Agentic AI', 'Fake News Detection', 'Multimodal AI', 'Tesseract OCR', 'Speech-to-Text', 'Real-time AI', 'Responsible AI'],
            featured: true,
            projectNumber: 72,
            totalProjects: 120,
            categoryProgress: '12/20 NLP Projects'
        },
        {
            id: 'nlp-13',
            title: 'Document Similarity Search using BERT Embeddings + FAISS',
            category: 'Natural Language Processing',
            domain: 'Information Retrieval & Semantic Search',
            description: 'Built a Document Similarity Search Engine using BERT embeddings to convert documents into semantic vectors with multiple search techniques for scalable retrieval.',
            image: 'port/nlp/13.jpg',
            video: 'port/nlp/13.mp4',
            technologies: ['Python', 'BERT', 'FAISS', 'Transformers', 'Cosine Similarity', 'Vector Search', 'Sentence Transformers', 'NumPy'],
            frameworks: ['FAISS', 'Sentence Transformers', 'Hugging Face'],
            accuracy: '92%',
            modelSize: '440MB',
            trainingTime: 'Pretrained (Indexing: 2 hours)',
            dataset: 'Legal Documents, Research Papers, Custom Document Corpus',
            keyFeatures: [
                'BERT embeddings for semantic document representation',
                'Multiple FAISS indexing strategies (Flat, HNSW, IVF)',
                'Cosine similarity for smaller datasets',
                'Lightning-fast retrieval on large document corpora',
                'Scalable vector search with sub-second query response',
                'Semantic search beyond keyword matching',
                'Batch document processing and indexing'
            ],
            technicalDetails: {
                embeddings: 'BERT/Sentence-BERT for 768-dimensional document vectors',
                indexing: 'FAISS Flat (exact), HNSW (approximate), IVF (clustered)',
                similarity: 'Cosine similarity and L2 distance metrics',
                scalability: 'Optimized for millions of documents',
                preprocessing: 'Document chunking and text normalization',
                evaluation: 'Retrieval accuracy, query speed, and relevance scoring'
            },
            results: {
                retrievalAccuracy: '92%',
                querySpeed: '< 100ms for 1M documents',
                indexingSpeed: '10K documents/minute',
                memoryEfficiency: '95% reduction vs brute force'
            },
            applications: [
                'Legal document retrieval systems',
                'Research paper recommendation engines',
                'Enterprise knowledge base search',
                'Content discovery and similarity matching'
            ],
            futurePlans: [
                'Implement hybrid search with keyword + semantic matching',
                'Add multilingual document support',
                'Deploy as enterprise search API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/document-similarity-bert-faiss',
            demoLink: 'https://document-search-demo.streamlit.app',
            tags: ['Document Search', 'BERT', 'FAISS', 'Semantic Search', 'Cosine Similarity', 'Vector Search', 'Information Retrieval'],
            featured: true,
            projectNumber: 73,
            totalProjects: 120,
            categoryProgress: '13/20 NLP Projects'
        },
        {
            id: 'nlp-14',
            title: 'Research Paper Summarizer using Agentic AI',
            category: 'Natural Language Processing',
            domain: 'Agentic AI & Academic Research',
            description: 'Built an intelligent Research Paper Summarizer powered by Agentic AI to automate the extraction of key insights from scientific documents with structured analysis.',
            image: 'port/nlp/14.jpg',
            video: 'port/nlp/14.mp4',
            technologies: ['Python', 'LangChain', 'OpenAI API', 'PyPDF2', 'Transformers', 'BERT', 'Streamlit', 'BeautifulSoup', 'Requests'],
            frameworks: ['LangChain', 'Transformers', 'Streamlit'],
            accuracy: '91%',
            modelSize: '750MB',
            trainingTime: 'Pretrained + Fine-tuning (4 hours)',
            dataset: 'ArXiv Papers, PubMed Articles, Custom Research Corpus',
            keyFeatures: [
                'Agentic AI workflow with specialized summarization agents',
                'Multi-format processing (PDF, text, URLs)',
                'Section-wise analysis (objectives, methodology, results, conclusions)',
                'Structured abstract generation with key insights extraction',
                'Language simplification for broader accessibility',
                'Document parsing with intelligent content extraction',
                'Real-time processing with progress tracking'
            ],
            technicalDetails: {
                agenticWorkflow: 'Multi-agent system with parsing, analysis, and summarization agents',
                documentParsing: 'PyPDF2 and BeautifulSoup for multi-format content extraction',
                sectionAnalysis: 'BERT-based classification for research paper sections',
                summarization: 'Transformer-based abstractive summarization',
                languageSimplification: 'GPT-based text simplification for accessibility',
                evaluation: 'ROUGE scores and expert evaluation metrics'
            },
            results: {
                summarizationAccuracy: '91%',
                sectionIdentification: '94%',
                processingSpeed: '2 minutes per 20-page paper',
                comprehensibilityScore: '88%'
            },
            applications: [
                'Academic research assistance',
                'Literature review automation',
                'Scientific knowledge extraction',
                'Educational content simplification'
            ],
            futurePlans: [
                'Add citation network analysis',
                'Implement multi-paper comparative summarization',
                'Deploy as academic research platform integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/research-paper-summarizer-agentic',
            demoLink: 'https://research-summarizer-demo.streamlit.app',
            tags: ['Agentic AI', 'Research Summarizer', 'Document Summarization', 'Scientific AI', 'Academic Research', 'LLM'],
            featured: true,
            projectNumber: 74,
            totalProjects: 120,
            categoryProgress: '14/20 NLP Projects'
        },
        {
            id: 'nlp-15',
            title: 'Resume Parser & Job Eligibility Checker using Gemini LLM',
            category: 'Natural Language Processing',
            domain: 'HR Technology & Recruitment AI',
            description: 'Developed a Resume Parser powered by Gemini LLM, capable of extracting candidate details and matching them against job requirements with intelligent eligibility analysis.',
            image: 'port/nlp/15.jpg',
            video: 'port/nlp/15.mp4',
            technologies: ['Python', 'Google Gemini API', 'Tesseract OCR', 'PyPDF2', 'Pillow', 'Streamlit', 'OpenCV', 'JSON'],
            frameworks: ['Google Gemini', 'Tesseract', 'Streamlit'],
            accuracy: '93%',
            modelSize: '2GB (Cloud-based)',
            trainingTime: 'Pretrained (API-based)',
            dataset: 'Resume Database, Job Descriptions, HR Requirements',
            keyFeatures: [
                'OCR-enabled resume processing (PDF/Image formats)',
                'Gemini LLM for intelligent content analysis',
                'Structured candidate profile extraction',
                'Real-time job requirement matching',
                'Role-specific fit analysis with natural language output',
                'Experience, skills, education, and certification parsing',
                'Career guidance and improvement recommendations'
            ],
            technicalDetails: {
                ocrProcessing: 'Tesseract OCR for image-based resume extraction',
                pdfParsing: 'PyPDF2 for text-based resume processing',
                llmAnalysis: 'Google Gemini for semantic understanding and matching',
                structuredOutput: 'JSON-formatted candidate profiles and eligibility scores',
                matching: 'Intelligent job requirement alignment with confidence scoring',
                evaluation: 'Parsing accuracy and job matching relevance metrics'
            },
            results: {
                parsingAccuracy: '93%',
                jobMatchingAccuracy: '89%',
                ocrAccuracy: '91%',
                processingSpeed: '15 seconds per resume'
            },
            applications: [
                'HR automation and recruitment systems',
                'Career guidance and counseling platforms',
                'Job recommendation engines',
                'Talent acquisition and screening tools'
            ],
            futurePlans: [
                'Add skill gap analysis and training recommendations',
                'Implement batch resume processing capabilities',
                'Deploy as enterprise HR platform integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/resume-parser-gemini-llm',
            demoLink: 'https://resume-parser-gemini-demo.streamlit.app',
            tags: ['Resume Parser', 'Gemini LLM', 'Job Eligibility', 'OCR', 'Tesseract', 'HR Tech', 'Intelligent Recruitment'],
            featured: true,
            projectNumber: 75,
            totalProjects: 120,
            categoryProgress: '15/20 NLP Projects'
        },
        {
            id: 'nlp-16',
            title: 'Voice Assistant using Whisper + Gemini + pyttsx3',
            category: 'Natural Language Processing',
            domain: 'Conversational AI & Voice Technology',
            description: 'Built a smart Voice Assistant by integrating multiple AI components to achieve full speech-to-speech communication using advanced NLP pipeline.',
            image: 'port/nlp/16.jpg',
            video: 'port/nlp/16.mp4',
            technologies: ['Python', 'OpenAI Whisper', 'Google Gemini API', 'pyttsx3', 'sounddevice', 'soundfile', 'Speech Recognition'],
            frameworks: ['OpenAI Whisper', 'Google Gemini', 'pyttsx3'],
            accuracy: '94%',
            modelSize: '550MB + Cloud API',
            trainingTime: 'Pretrained (Real-time inference)',
            dataset: 'Real-time Voice Input, Conversational Data',
            keyFeatures: [
                'Complete speech-to-speech communication pipeline',
                'Real-time voice recording using sounddevice',
                'High-accuracy speech-to-text with OpenAI Whisper',
                'Intelligent response generation using Gemini LLM',
                'Natural text-to-speech conversion with pyttsx3',
                'Continuous conversation flow with context awareness',
                'Multi-platform voice interaction support'
            ],
            technicalDetails: {
                voiceRecording: 'sounddevice for real-time audio capture and soundfile for storage',
                speechToText: 'OpenAI Whisper for robust speech recognition',
                llmProcessing: 'Google Gemini API for intelligent response generation',
                textToSpeech: 'pyttsx3 for natural voice synthesis',
                pipeline: 'Seamless audio → text → LLM → text → audio workflow',
                evaluation: 'End-to-end conversation quality and response accuracy'
            },
            results: {
                speechRecognitionAccuracy: '94%',
                responseRelevance: '91%',
                voiceSynthesisQuality: '89%',
                conversationLatency: '3-5 seconds per interaction'
            },
            applications: [
                'Smart home voice assistants',
                'Accessibility tools for visually impaired',
                'Customer service automation bots',
                'Educational interactive tutoring systems'
            ],
            futurePlans: [
                'Add emotion detection and response adaptation',
                'Implement wake word detection for hands-free activation',
                'Deploy on edge devices for offline functionality'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/voice-assistant-whisper-gemini',
            demoLink: 'https://voice-assistant-demo.streamlit.app',
            tags: ['Voice Assistant', 'Whisper', 'Gemini LLM', 'Speech-to-Text', 'Text-to-Speech', 'pyttsx3', 'Conversational AI'],
            featured: true,
            projectNumber: 76,
            totalProjects: 120,
            categoryProgress: '16/20 NLP Projects'
        },
        {
            id: 'nlp-17',
            title: 'Sequence Classification on IMDB using LSTM',
            category: 'Natural Language Processing',
            domain: 'Sentiment Analysis & Sequential Learning',
            description: 'Developed a Sequence Classification model using LSTM (Long Short-Term Memory) to classify IMDB movie reviews into positive or negative sentiments with deep sequential understanding.',
            image: 'port/nlp/17.jpg',
            video: 'port/nlp/17.mp4',
            technologies: ['Python', 'TensorFlow', 'Keras', 'LSTM', 'Embedding Layer', 'Tokenization', 'Padding', 'NumPy'],
            frameworks: ['TensorFlow', 'Keras'],
            accuracy: '87%',
            modelSize: '25MB',
            trainingTime: '2.5 hours',
            dataset: 'IMDB Movie Reviews Dataset (50K reviews)',
            keyFeatures: [
                'LSTM-based sequential sentiment classification',
                'Comprehensive text preprocessing with tokenization and padding',
                'Embedding layer for word representation learning',
                'Binary classification for positive/negative sentiment',
                'Validation split with epoch-wise performance monitoring',
                'Sequential model architecture with dropout regularization',
                'Real-time sentiment prediction capability'
            ],
            technicalDetails: {
                architecture: 'Embedding + LSTM + Dense layers with dropout',
                preprocessing: 'Tokenization, sequence padding, and vocabulary mapping',
                embedding: 'Trainable word embeddings for semantic representation',
                lstm: 'Long Short-Term Memory for sequential pattern learning',
                training: 'Binary cross-entropy loss with Adam optimizer',
                evaluation: 'Accuracy, loss curves, and validation performance tracking'
            },
            results: {
                accuracy: '87%',
                validationAccuracy: '85%',
                precision: '86%',
                recall: '88%',
                f1Score: '87%'
            },
            applications: [
                'Movie review sentiment analysis',
                'Product review classification systems',
                'Social media sentiment monitoring',
                'Customer feedback analysis platforms'
            ],
            futurePlans: [
                'Implement Bidirectional LSTM for improved context',
                'Compare with GRU and Transformer architectures',
                'Deploy as real-time sentiment analysis API'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/imdb-lstm-sequence-classification',
            demoLink: 'https://imdb-lstm-sentiment-demo.streamlit.app',
            tags: ['IMDB', 'LSTM', 'Sequence Classification', 'Sentiment Analysis', 'RNN', 'Text Classification'],
            featured: true,
            projectNumber: 77,
            totalProjects: 120,
            categoryProgress: '17/20 NLP Projects'
        },
        {
            id: 'nlp-18',
            title: 'Emotion Classification using DistilRoBERTa + Gradio App',
            category: 'Natural Language Processing',
            domain: 'Emotion AI & Affective Computing',
            description: 'Built an Emotion Detection System using the lightweight yet powerful DistilRoBERTa model to classify text into emotions like joy, anger, sadness, fear, and more.',
            image: 'port/nlp/18.jpg',
            video: 'port/nlp/18.mp4',
            technologies: ['Python', 'DistilRoBERTa', 'Transformers', 'Gradio', 'Hugging Face', 'PyTorch', 'Tokenization'],
            frameworks: ['Hugging Face Transformers', 'Gradio'],
            accuracy: '92%',
            modelSize: '82MB',
            trainingTime: 'Pretrained + Fine-tuning (2 hours)',
            dataset: 'Emotion Dataset (Joy, Anger, Sadness, Fear, Surprise, Love)',
            keyFeatures: [
                'DistilRoBERTa for efficient emotion classification',
                'Multi-class emotion detection (6+ emotion categories)',
                'Interactive Gradio web interface for real-time testing',
                'Lightweight model with fast inference speed',
                'Advanced text preprocessing with tokenization and padding',
                'Real-time emotion analysis for social media and feedback',
                'Confidence scoring for emotion predictions'
            ],
            technicalDetails: {
                architecture: 'DistilRoBERTa-base with emotion classification head',
                preprocessing: 'Tokenization, truncation, and attention mask generation',
                emotions: 'Joy, Anger, Sadness, Fear, Surprise, Love, Neutral',
                interface: 'Gradio web app with text input and emotion output',
                optimization: 'Distilled model for 60% faster inference than RoBERTa',
                evaluation: 'Multi-class accuracy, F1-score, and confusion matrix'
            },
            results: {
                overallAccuracy: '92%',
                joyDetection: '94%',
                angerDetection: '91%',
                sadnessDetection: '90%',
                inferenceSpeed: '50ms per text'
            },
            applications: [
                'Social media emotion monitoring',
                'Customer feedback sentiment analysis',
                'Mental health assessment tools',
                'Chatbot emotional intelligence enhancement'
            ],
            futurePlans: [
                'Add multilingual emotion detection support',
                'Implement real-time streaming emotion analysis',
                'Deploy as emotion analytics API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/emotion-classification-distilroberta',
            demoLink: 'https://emotion-detector-gradio-demo.huggingface.co',
            tags: ['Emotion Classification', 'DistilRoBERTa', 'Transformers', 'Gradio', 'Emotion AI', 'Affective Computing'],
            featured: true,
            projectNumber: 78,
            totalProjects: 120,
            categoryProgress: '18/20 NLP Projects'
        },
        {
            id: 'nlp-19',
            title: 'Corona Research Agent using Agentic AI + Gemma 2 9B-IT',
            category: 'Natural Language Processing',
            domain: 'Agentic AI & Healthcare Research',
            description: 'Developed a Corona Research Agent that autonomously retrieves, analyzes, and summarizes COVID-19 related data using Agentic AI and Gemma 2 9B-IT for healthcare research automation.',
            image: 'port/nlp/19.jpg',
            video: 'port/nlp/19.mp4',
            technologies: ['Python', 'Gemma 2 9B-IT', 'LangChain', 'Google Search API', 'BeautifulSoup', 'Streamlit', 'Transformers'],
            frameworks: ['Gemma 2', 'LangChain', 'Streamlit'],
            accuracy: '90%',
            modelSize: '18GB',
            trainingTime: 'Pretrained + Fine-tuning (6 hours)',
            dataset: 'COVID-19 Research Papers, WHO Data, PubMed Articles',
            keyFeatures: [
                'Gemma 2 9B-IT for domain-specific healthcare understanding',
                'Autonomous agentic workflow for research automation',
                'Google Search Tool integration for live data retrieval',
                'Real-time COVID-19 research article analysis',
                'Intelligent document parsing and summarization',
                'Interactive Streamlit interface for user queries',
                'Multi-source data aggregation and synthesis'
            ],
            technicalDetails: {
                agenticWorkflow: 'Query → Web Search → Document Parsing → Analysis → Summarization',
                llmEngine: 'Gemma 2 9B-IT for healthcare-specific language understanding',
                searchIntegration: 'Google Search API for real-time research retrieval',
                documentParsing: 'BeautifulSoup and custom parsers for research papers',
                interface: 'Streamlit web app with interactive query interface',
                evaluation: 'Research relevance, summary accuracy, and source credibility'
            },
            results: {
                researchAccuracy: '90%',
                summaryQuality: '88%',
                sourceRelevance: '92%',
                queryResponseTime: '30-45 seconds'
            },
            applications: [
                'Healthcare research automation',
                'Medical literature review assistance',
                'Public health policy research',
                'Academic COVID-19 study support'
            ],
            futurePlans: [
                'Expand to other disease research domains',
                'Add citation network analysis capabilities',
                'Deploy as healthcare research platform integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/corona-research-agent-gemma2',
            demoLink: 'https://corona-research-agent-demo.streamlit.app',
            tags: ['Agentic AI', 'Gemma 2', 'Corona Research', 'Healthcare AI', 'Research Automation', 'COVID-19 AI'],
            featured: true,
            projectNumber: 79,
            totalProjects: 120,
            categoryProgress: '19/20 NLP Projects'
        },
        {
            id: 'nlp-20',
            title: 'Text Generation using Falcon Model (Hugging Face)',
            category: 'Natural Language Processing',
            domain: 'Large Language Models & Text Generation',
            description: 'Built a powerful Text Generation system using Falcon LLM, a high-performing open-source language model available on Hugging Face for creative content automation.',
            image: 'port/nlp/20.jpg',
            video: 'port/nlp/20.mp4',
            technologies: ['Python', 'Falcon-7B', 'Hugging Face', 'Transformers', 'PyTorch', 'Tokenizers', 'GPU Acceleration'],
            frameworks: ['Hugging Face Transformers', 'PyTorch'],
            accuracy: '89%',
            modelSize: '14GB',
            trainingTime: 'Pretrained (Inference optimization: 3 hours)',
            dataset: 'RefinedWeb Dataset, Custom Creative Writing Corpus',
            keyFeatures: [
                'Falcon-7B large language model for high-quality text generation',
                'Creative writing capabilities (stories, blogs, articles)',
                'Code completion and programming assistance',
                'Zero-shot and few-shot learning across multiple domains',
                'Hugging Face Transformers integration for seamless deployment',
                'GPU-accelerated inference for fast generation',
                'Customizable generation parameters (temperature, top-k, top-p)'
            ],
            technicalDetails: {
                architecture: 'Falcon-7B transformer with 7 billion parameters',
                generation: 'Autoregressive text generation with nucleus sampling',
                optimization: 'GPU acceleration with mixed precision inference',
                prompting: 'Zero-shot and few-shot prompt engineering',
                parameters: 'Configurable temperature, top-k, and top-p sampling',
                evaluation: 'Perplexity, coherence, and creativity assessment'
            },
            results: {
                textQuality: '89%',
                coherenceScore: '91%',
                creativityRating: '87%',
                generationSpeed: '25 tokens/second'
            },
            applications: [
                'Creative writing and storytelling assistance',
                'Blog and article content generation',
                'Code completion and programming help',
                'Educational content creation and tutoring'
            ],
            futurePlans: [
                'Fine-tune on domain-specific datasets',
                'Implement instruction-following capabilities',
                'Deploy as content creation API service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/text-generation-falcon-llm',
            demoLink: 'https://falcon-text-generator-demo.streamlit.app',
            tags: ['Falcon', 'Text Generation', 'Hugging Face', 'LLM', 'Open Source AI', 'Generative AI'],
            featured: true,
            projectNumber: 80,
            totalProjects: 120,
            categoryProgress: '20/20 NLP Projects - COMPLETED! 🎉'
        }
    ],

    // 🎨 Generative AI Projects
    genai: [
        {
            id: 'genai-1',
            title: 'Visual LLM Agent Studio using LangFlow + LangChain + FastAPI',
            category: 'Generative AI',
            domain: 'LLM Orchestration & Visual Development',
            description: 'Built a Visual LLM Agent Studio powered by LangFlow, LangChain, and a custom backend with FastAPI for drag-and-drop LLM workflow creation and deployment.',
            image: 'port/genai/1.jpg',
            video: 'port/genai/1.mp4',
            technologies: ['Python', 'LangFlow', 'LangChain', 'FastAPI', 'Uvicorn', 'React', 'JavaScript', 'HTML/CSS'],
            frameworks: ['LangFlow', 'LangChain', 'FastAPI'],
            accuracy: '95%',
            modelSize: 'Variable (depends on LLM selection)',
            trainingTime: 'No training (Visual workflow builder)',
            dataset: 'Custom LLM Workflows and Agent Configurations',
            keyFeatures: [
                'Drag-and-drop visual LLM workflow designer using LangFlow',
                'Custom frontend integration via iframe with LangFlow GitHub code',
                'Automatic backend export and deployment with FastAPI + Uvicorn',
                'LangChain orchestration for intelligent agent pipelines',
                'Rapid prototyping environment for LLM applications',
                'No-code/low-code interface for non-technical users',
                'Real-time workflow testing and debugging capabilities'
            ],
            technicalDetails: {
                frontend: 'LangFlow visual interface with custom React components',
                backend: 'FastAPI with automatic workflow export and deployment',
                orchestration: 'LangChain for agent pipeline management and execution',
                deployment: 'Uvicorn ASGI server for production-ready API endpoints',
                integration: 'Iframe embedding for seamless frontend-backend communication',
                evaluation: 'Workflow execution success rate and performance metrics'
            },
            results: {
                workflowSuccess: '95%',
                deploymentSpeed: '< 30 seconds',
                userAdoption: '92% ease of use rating',
                apiResponseTime: '200ms average'
            },
            applications: [
                'Chatbot and conversational AI development',
                'Research agent and automation pipeline creation',
                'Content generation workflow automation',
                'Educational LLM experimentation platform'
            ],
            futurePlans: [
                'Add support for multimodal LLM workflows',
                'Implement collaborative workflow sharing',
                'Deploy as cloud-based SaaS platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/visual-llm-agent-studio',
            demoLink: 'https://visual-llm-studio-demo.streamlit.app',
            tags: ['LangFlow', 'LangChain', 'FastAPI', 'Visual LLM', 'No-Code AI', 'Agent Studio', 'LLM Orchestration'],
            featured: true,
            projectNumber: 81,
            totalProjects: 120,
            categoryProgress: '1/20 GenAI Projects'
        },
        {
            id: 'genai-2',
            title: 'InsightServe: Chain-to-API System using LangServe + LangChain + FastAPI + LangSmith',
            category: 'Generative AI',
            domain: 'LLM Infrastructure & API Development',
            description: 'Developed InsightServe, a powerful Chain-to-API system that transforms LLM chains into deployable APIs with comprehensive monitoring and analytics capabilities.',
            image: 'port/genai/2.jpg',
            video: 'port/genai/2.mp4',
            technologies: ['Python', 'LangServe', 'LangChain', 'LangSmith', 'FastAPI', 'Uvicorn', 'Streamlit', 'RESTful APIs'],
            frameworks: ['LangServe', 'LangChain', 'LangSmith', 'FastAPI', 'Streamlit'],
            accuracy: '96%',
            modelSize: 'Variable (chain-dependent)',
            trainingTime: 'No training (Infrastructure system)',
            dataset: 'Custom LLM Chains and API Configurations',
            keyFeatures: [
                'LangChain for modular LLM pipeline construction',
                'LangServe for seamless chain-to-API transformation',
                'FastAPI + Uvicorn for robust backend deployment',
                'LangSmith integration for real-time debugging and observability',
                'Performance logging and analytics dashboard',
                'Streamlit frontend for interactive chain testing',
                'Enterprise-grade scalability and monitoring'
            ],
            technicalDetails: {
                chainBuilding: 'LangChain modular pipeline architecture',
                apiDeployment: 'LangServe automatic RESTful API generation',
                backend: 'FastAPI with Uvicorn ASGI server for production deployment',
                monitoring: 'LangSmith real-time observability and performance tracking',
                frontend: 'Streamlit interactive interface for chain testing',
                evaluation: 'API response time, chain execution success, and system reliability'
            },
            results: {
                apiReliability: '96%',
                averageResponseTime: '150ms',
                chainExecutionSuccess: '94%',
                systemUptime: '99.2%'
            },
            applications: [
                'Enterprise LLM service deployment',
                'Rapid AI prototype to production pipeline',
                'Custom AI tool scaling and distribution',
                'LLM chain monitoring and optimization'
            ],
            futurePlans: [
                'Add auto-scaling capabilities for high-traffic scenarios',
                'Implement A/B testing framework for chain optimization',
                'Deploy as cloud-native microservices architecture'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/insightserve-chain-to-api',
            demoLink: 'https://insightserve-demo.streamlit.app',
            tags: ['InsightServe', 'LangServe', 'LangChain', 'LangSmith', 'Chain-to-API', 'AI Infrastructure', 'Enterprise AI'],
            featured: true,
            projectNumber: 82,
            totalProjects: 120,
            categoryProgress: '2/20 GenAI Projects'
        },
        {
            id: 'genai-3',
            title: 'LLM Debug Lab: Real-Time Evaluation Dashboard using LangSmith + LangChain + Gradio',
            category: 'Generative AI',
            domain: 'LLM Debugging & Performance Evaluation',
            description: 'Built LLM Debug Lab, a real-time interactive environment for evaluating and debugging LLM chains, leveraging the LangSmith ecosystem for comprehensive observability.',
            image: 'port/genai/3.jpg',
            video: 'port/genai/3.mp4',
            technologies: ['Python', 'LangSmith', 'LangChain', 'Gradio', 'Performance Monitoring', 'Error Logging', 'Real-time Analytics'],
            frameworks: ['LangSmith', 'LangChain', 'Gradio'],
            accuracy: '97%',
            modelSize: 'Variable (chain-dependent)',
            trainingTime: 'No training (Debugging environment)',
            dataset: 'LLM Chain Execution Logs and Performance Metrics',
            keyFeatures: [
                'Real-time LLM chain debugging and evaluation environment',
                'LangChain integration for modular pipeline construction',
                'LangSmith monitoring for inputs, outputs, and intermediate steps',
                'Interactive Gradio chat interface for live testing',
                'Detailed performance tracking and analytics dashboard',
                'Agent feedback collection and error logging system',
                'Transparent LLM pipeline optimization tools'
            ],
            technicalDetails: {
                chainDebugging: 'LangChain pipeline construction with step-by-step monitoring',
                observability: 'LangSmith real-time tracking of chain execution flow',
                interface: 'Gradio interactive chat for immediate feedback and testing',
                monitoring: 'Performance metrics, latency tracking, and error analysis',
                logging: 'Comprehensive error logging and debugging information',
                evaluation: 'Chain performance assessment and optimization recommendations'
            },
            results: {
                debuggingAccuracy: '97%',
                performanceInsights: '94% issue detection rate',
                userProductivity: '85% faster debugging cycles',
                systemReliability: '98.5% uptime'
            },
            applications: [
                'LLM pipeline development and optimization',
                'AI research and experimentation platforms',
                'Production LLM system monitoring',
                'Educational AI debugging and learning tools'
            ],
            futurePlans: [
                'Add automated performance optimization suggestions',
                'Implement collaborative debugging features',
                'Deploy as cloud-based debugging service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/llm-debug-lab-langsmith',
            demoLink: 'https://llm-debug-lab-demo.gradio.app',
            tags: ['LLM Debug Lab', 'LangSmith', 'LangChain', 'Gradio', 'Observability', 'AI Engineering', 'Performance Monitoring'],
            featured: true,
            projectNumber: 83,
            totalProjects: 120,
            categoryProgress: '3/20 GenAI Projects'
        },
        {
            id: 'genai-5',
            title: 'LangAudit: LLM Evaluation with Hallucination Scoring using LangSmith + LangChain + CSV Logger',
            category: 'Generative AI',
            domain: 'AI Safety & Model Quality Assurance',
            description: 'Developed LangAudit, a comprehensive LLM auditing framework that detects and evaluates hallucinations in LLM outputs with advanced scoring and logging capabilities.',
            image: 'port/genai/5.jpg',
            video: 'port/genai/5.mp4',
            technologies: ['Python', 'LangSmith', 'LangChain', 'CSV Logger', 'Hallucination Detection', 'Statistical Analysis', 'Data Validation'],
            frameworks: ['LangSmith', 'LangChain', 'Pandas'],
            accuracy: '94%',
            modelSize: 'Variable (evaluation framework)',
            trainingTime: 'No training (Evaluation system)',
            dataset: 'LLM Output Samples, Ground Truth Data, Hallucination Benchmarks',
            keyFeatures: [
                'Comprehensive LLM auditing and evaluation framework',
                'Custom Hallucination Score system for confidence assessment',
                'LangChain integration for structured LLM chain testing',
                'LangSmith tracking for prompts, outputs, and metadata analysis',
                'CSV Logger for structured data export and reproducibility',
                'Automated flagging of low-confidence or fabricated responses',
                'Statistical analysis and quality assurance reporting'
            ],
            technicalDetails: {
                hallucinationDetection: 'Custom scoring algorithm based on confidence, consistency, and factual accuracy',
                chainTesting: 'LangChain structured evaluation of various LLM configurations',
                monitoring: 'LangSmith comprehensive tracking of evaluation metrics and metadata',
                logging: 'CSV-based structured logging for analytics and audit trails',
                scoring: 'Multi-dimensional hallucination scoring (0-100 scale)',
                evaluation: 'Precision, recall, and F1-score for hallucination detection accuracy'
            },
            results: {
                hallucinationDetection: '94%',
                falsePositiveRate: '6%',
                auditAccuracy: '92%',
                processingSpeed: '500 evaluations/minute'
            },
            applications: [
                'AI safety and compliance monitoring',
                'Model quality assurance workflows',
                'Production LLM output validation',
                'Research and development evaluation pipelines'
            ],
            futurePlans: [
                'Implement real-time hallucination detection API',
                'Add multi-modal hallucination detection capabilities',
                'Deploy as enterprise AI governance platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/langaudit-hallucination-detection',
            demoLink: 'https://langaudit-demo.streamlit.app',
            tags: ['LangAudit', 'Hallucination Detection', 'LLM Evaluation', 'AI Safety', 'Model Auditing', 'Quality Assurance'],
            featured: true,
            projectNumber: 85,
            totalProjects: 120,
            categoryProgress: '5/20 GenAI Projects'
        },
        {
            id: 'genai-6',
            title: 'TaskAgents: Auto-Coordinating AI Team using CrewAI + AutoGen',
            category: 'Generative AI',
            domain: 'Multi-Agent Systems & AI Orchestration',
            description: 'Built TaskAgents, a system of autonomous, goal-oriented AI agents that collaborate and self-coordinate to solve complex tasks using CrewAI and AutoGen frameworks.',
            image: 'port/genai/6.jpg',
            video: 'port/genai/6.mp4',
            technologies: ['Python', 'CrewAI', 'AutoGen', 'LLMs', 'Search APIs', 'File Processing', 'Multi-Agent Systems'],
            frameworks: ['CrewAI', 'AutoGen', 'LangChain'],
            accuracy: '91%',
            modelSize: 'Variable (multi-agent system)',
            trainingTime: 'No training (Agent orchestration)',
            dataset: 'Task Definitions, Agent Workflows, Collaboration Patterns',
            keyFeatures: [
                'Autonomous AI agent team with self-coordination capabilities',
                'CrewAI for role definition and dynamic workflow management',
                'AutoGen for message-passing and inter-agent communication',
                'Multi-agent tool integration (LLMs, search APIs, file readers)',
                'Self-reflection and adaptive task execution',
                'Complex task breakdown and parallel processing',
                'Intelligent agent-based problem solving'
            ],
            technicalDetails: {
                agentOrchestration: 'CrewAI role-based agent coordination and workflow management',
                communication: 'AutoGen message-passing and tool calling between agents',
                toolIntegration: 'LLMs, web search APIs, document processors, and custom tools',
                taskExecution: 'Parallel and sequential task breakdown with self-coordination',
                reflection: 'Agent self-assessment and adaptive behavior modification',
                evaluation: 'Task completion rate, collaboration efficiency, and output quality'
            },
            results: {
                taskCompletionRate: '91%',
                collaborationEfficiency: '88%',
                outputQuality: '89%',
                processingSpeed: '3x faster than single-agent systems'
            },
            applications: [
                'Automated research and analysis workflows',
                'Multi-step document processing and generation',
                'Code development and review automation',
                'Complex problem-solving and decision support'
            ],
            futurePlans: [
                'Add specialized domain agents (finance, healthcare, legal)',
                'Implement hierarchical agent management systems',
                'Deploy as enterprise workflow automation platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/taskagents-crewai-autogen',
            demoLink: 'https://taskagents-demo.streamlit.app',
            tags: ['TaskAgents', 'CrewAI', 'AutoGen', 'Multi-Agent AI', 'Autonomous Agents', 'AI Orchestration', 'Collaborative AI'],
            featured: true,
            projectNumber: 86,
            totalProjects: 120,
            categoryProgress: '6/20 GenAI Projects'
        },
        {
            id: 'genai-7',
            title: 'StateChatAI: Conversational State Manager using LangGraph + AutoGen + LLaMA3-8B',
            category: 'Generative AI',
            domain: 'Conversational AI & State Management',
            description: 'Developed StateChatAI, an intelligent conversational state manager that tracks and manages multi-turn conversations using advanced state machine architecture.',
            image: 'port/genai/7.jpg',
            video: 'port/genai/7.mp4',
            technologies: ['Python', 'LangGraph', 'AutoGen', 'LLaMA3-8B-8192', 'State Machines', 'Memory Management', 'Context Tracking'],
            frameworks: ['LangGraph', 'AutoGen', 'LLaMA3'],
            accuracy: '93%',
            modelSize: '16GB (LLaMA3-8B)',
            trainingTime: 'Pretrained + State Configuration (4 hours)',
            dataset: 'Multi-turn Conversation Logs, Dialog State Annotations',
            keyFeatures: [
                'Event-driven conversational state machine using LangGraph',
                'AutoGen integration for message flow and reasoning loops',
                'LLaMA3-8B-8192 for context-aware dialog generation',
                'Custom nodes for memory, context transition, and goal completion',
                'Memory-aware conversations with persistent context tracking',
                'Goal-driven interactions with dynamic state transitions',
                'Multi-turn agent coordination in long-form dialogues'
            ],
            technicalDetails: {
                stateMachine: 'LangGraph event-driven state management with custom node architecture',
                messageFlow: 'AutoGen reasoning loops and function calling for dialog coordination',
                llmEngine: 'LLaMA3-8B-8192 for context-aware response generation',
                memorySystem: 'Persistent conversation memory with context window management',
                stateTransition: 'Dynamic state transitions based on conversation goals and context',
                evaluation: 'Conversation coherence, goal completion rate, and state accuracy'
            },
            results: {
                conversationCoherence: '93%',
                goalCompletionRate: '89%',
                stateAccuracy: '91%',
                contextRetention: '94%'
            },
            applications: [
                'Customer service chatbots with memory',
                'AI tutoring systems with learning progression',
                'Planning and scheduling assistants',
                'Multi-turn agent workflow coordination'
            ],
            futurePlans: [
                'Add emotional state tracking and adaptation',
                'Implement multi-user conversation state management',
                'Deploy as conversational AI platform service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/statechatai-langgraph-autogen',
            demoLink: 'https://statechatai-demo.streamlit.app',
            tags: ['StateChatAI', 'LangGraph', 'AutoGen', 'LLaMA3', 'Conversational AI', 'State Management', 'Memory-Aware AI'],
            featured: true,
            projectNumber: 87,
            totalProjects: 120,
            categoryProgress: '7/20 GenAI Projects'
        },
        {
            id: 'genai-8',
            title: 'DecisionSim: Business Decision Simulator using LangGraph + AutoGen + CrewAI',
            category: 'Generative AI',
            domain: 'Business Intelligence & Decision Simulation',
            description: 'Built DecisionSim, a multi-agent decision-making simulator designed to model real-world business strategy sessions with specialized analyst agents and collaborative workflows.',
            image: 'port/genai/8.jpg',
            video: 'port/genai/8.mp4',
            technologies: ['Python', 'LangGraph', 'AutoGen', 'CrewAI', 'LLaMA3-8B-8192', 'DeepSeek R1', 'Gradio', 'Business Intelligence'],
            frameworks: ['LangGraph', 'AutoGen', 'CrewAI', 'Gradio'],
            accuracy: '92%',
            modelSize: '16GB + Cloud API',
            trainingTime: 'Pretrained + Agent Configuration (5 hours)',
            dataset: 'Business Case Studies, Strategic Decision Scenarios, Market Data',
            keyFeatures: [
                'Multi-agent business strategy simulation with specialized roles',
                'Market Analyst agent for data-driven insights and analysis',
                'Finance Analyst agent for risk assessment and cost evaluation',
                'CEO Analyst agent for final strategic decision making',
                'LangGraph state transitions for multi-step agent deliberation',
                'AutoGen message flow control and feedback loops',
                'CrewAI agent role coordination and structured workflows',
                'Gradio interface for intuitive simulation interaction'
            ],
            technicalDetails: {
                agentRoles: 'Market Analyst (insights), Finance Analyst (risk), CEO Analyst (decisions)',
                stateManagement: 'LangGraph multi-step deliberation with state transitions',
                messageFlow: 'AutoGen controlled communication and feedback loops',
                coordination: 'CrewAI structured workflows and role-based collaboration',
                llmEngines: 'LLaMA3-8B-8192 for strategic reasoning, DeepSeek R1 for analysis',
                evaluation: 'Decision quality, reasoning transparency, and simulation accuracy'
            },
            results: {
                decisionAccuracy: '92%',
                reasoningTransparency: '89%',
                simulationRealism: '90%',
                collaborationEfficiency: '87%'
            },
            applications: [
                'Product launch strategy simulation',
                'Investment planning and risk assessment',
                'Competitive response strategy development',
                'Business scenario planning and analysis'
            ],
            futurePlans: [
                'Add industry-specific agent specializations',
                'Implement real-time market data integration',
                'Deploy as enterprise decision support platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/decisionsim-business-simulator',
            demoLink: 'https://decisionsim-demo.gradio.app',
            tags: ['DecisionSim', 'Business AI', 'LangGraph', 'AutoGen', 'CrewAI', 'Strategic Planning', 'Multi-Agent Simulation'],
            featured: true,
            projectNumber: 88,
            totalProjects: 120,
            categoryProgress: '8/20 GenAI Projects'
        },
        {
            id: 'genai-9',
            title: 'ProjectPilot: Multi-Agent AI Orchestration Framework',
            category: 'Generative AI',
            domain: 'Multi-Agent Orchestration & Task Automation',
            description: 'ProjectPilot is a full-stack, modular multi-agent orchestration system built for complex task execution pipelines with complete lifecycle management.',
            image: 'port/genai/9.jpg',
            video: 'port/genai/9.mp4',
            technologies: ['Python', 'LangGraph', 'CrewAI', 'Poetry', 'DeepSeek R1', 'LLaMA 70B', 'pyproject.toml', 'Logging System'],
            frameworks: ['LangGraph', 'CrewAI', 'Poetry'],
            accuracy: '94%',
            modelSize: '140GB (LLaMA 70B) + Cloud API',
            trainingTime: 'Pretrained + Framework Configuration (6 hours)',
            dataset: 'Task Execution Pipelines, Agent Workflow Patterns, Documentation Corpus',
            keyFeatures: [
                'Full-stack modular multi-agent orchestration system',
                'LangGraph flow-based task routing and agent state control',
                'CrewAI role-based agent behavior and collaboration',
                'Poetry local Python packaging and dependency management',
                'Complete agent lifecycle orchestration with specialized roles',
                'Intake Agent for initial user task prompt handling',
                'Data Fetcher for web/API-based context retrieval',
                'Planner for dynamic workflow structuring',
                'Writer for report/content/code generation',
                'Reviewer for output quality validation'
            ],
            technicalDetails: {
                orchestration: 'LangGraph flow-based routing with CrewAI role coordination',
                agentRoles: 'Intake, Data Fetcher, Planner, Writer, Reviewer agents',
                packageManagement: 'Poetry with pyproject.toml for local development',
                llmEngines: 'DeepSeek R1 for analysis, LLaMA 70B for high-context reasoning',
                logging: 'Comprehensive activity logs and error tracing system',
                evaluation: 'Task completion rate, output quality, and workflow efficiency'
            },
            results: {
                taskCompletionRate: '94%',
                workflowEfficiency: '91%',
                outputQuality: '93%',
                systemReliability: '96%'
            },
            applications: [
                'Automated document generation pipelines',
                'Research workflow automation',
                'Content creation and review systems',
                'Enterprise task orchestration platforms'
            ],
            futurePlans: [
                'Add cloud-native deployment capabilities',
                'Implement real-time collaboration features',
                'Deploy as enterprise workflow automation platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/projectpilot-orchestration-framework',
            demoLink: 'https://projectpilot-demo.streamlit.app',
            tags: ['ProjectPilot', 'LangGraph', 'CrewAI', 'Poetry', 'Multi-Agent Orchestration', 'Task Automation', 'LLaMA 70B'],
            featured: true,
            projectNumber: 89,
            totalProjects: 120,
            categoryProgress: '9/20 GenAI Projects'
        },
        {
            id: 'genai-10',
            title: 'AgentAudit: Multi-Agent Evaluation System',
            category: 'Generative AI',
            domain: 'Agent Performance Evaluation & Quality Assurance',
            description: 'AgentAudit is a robust agent performance evaluation framework combining Autogen for dynamic agent orchestration and LangSmith for comprehensive tracing, logging, and monitoring.',
            image: 'port/genai/10.jpg',
            video: 'port/genai/10.mp4',
            technologies: ['Python', 'Autogen', 'LangSmith', 'CSV Logger', 'Performance Monitoring', 'Hallucination Detection', 'Quality Assurance'],
            frameworks: ['Autogen', 'LangSmith', 'Pandas'],
            accuracy: '95%',
            modelSize: 'Variable (configurable LLMs)',
            trainingTime: 'No training (Evaluation framework)',
            dataset: 'Agent Task Executions, Performance Benchmarks, Quality Metrics',
            keyFeatures: [
                'Dynamic multi-agent orchestration using Autogen',
                'Working Agent for real-world task execution',
                'Meta-Evaluator Agent for output critique and consistency checking',
                'LangSmith integration for comprehensive tracing and logging',
                'Hallucination score tracking and error monitoring',
                'Task-to-task evaluation pipeline with score-based feedback',
                'Automated CSV export for audit trails and analysis'
            ],
            technicalDetails: {
                agentOrchestration: 'Autogen dynamic agent coordination and task distribution',
                workingAgent: 'Real-world task execution with performance tracking',
                metaEvaluator: 'Output evaluation, inconsistency flagging, and critique generation',
                monitoring: 'LangSmith tracing, logging, and hallucination detection',
                evaluation: 'Task-level scorecards with timestamped metadata',
                reporting: 'Automated CSV logging and performance analytics'
            },
            results: {
                evaluationAccuracy: '95%',
                hallucinationDetection: '92%',
                taskCompletionTracking: '97%',
                auditTrailCompleteness: '99%'
            },
            applications: [
                'AI research lab quality assurance',
                'LLM operations and reliability monitoring',
                'Agent benchmarking and performance optimization',
                'Production AI system validation'
            ],
            futurePlans: [
                'Add real-time dashboard for agent performance monitoring',
                'Implement automated agent improvement recommendations',
                'Deploy as enterprise AI governance platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/agentaudit-evaluation-system',
            demoLink: 'https://agentaudit-demo.streamlit.app',
            tags: ['AgentAudit', 'Autogen', 'LangSmith', 'Meta-Agent', 'AI Evaluation', 'Hallucination Tracking', 'Quality Assurance'],
            featured: true,
            projectNumber: 90,
            totalProjects: 120,
            categoryProgress: '10/20 GenAI Projects'
        },
        {
            id: 'genai-11',
            title: 'DevBotAppAI: Full-Stack Dev Agent with Docker + GitHub + SERPAPI + Taipy',
            category: 'Generative AI',
            domain: 'AI-Powered Development & DevOps Automation',
            description: 'DevBotAppAI is a complete AI-powered development assistant that leverages LangChain Agent Tools, Docker integration, and real-time web access via SerpAPI, combined with a Taipy-generated frontend.',
            image: 'port/genai/11.jpg',
            video: 'port/genai/11.mp4',
            technologies: ['Python', 'LangChain', 'Docker SDK', 'GitHub REST API', 'SerpAPI', 'Taipy', 'Custom Agent Tools', 'Real-time Streaming'],
            frameworks: ['LangChain', 'Docker', 'Taipy', 'GitHub API'],
            accuracy: '93%',
            modelSize: 'Variable (LangChain agent-dependent)',
            trainingTime: 'No training (Tool-based agent)',
            dataset: 'GitHub Repositories, Docker Containers, Real-time Web Data',
            keyFeatures: [
                'Complete AI-powered development assistant with multi-tool integration',
                'LangChain custom agent with Docker CLI interaction capabilities',
                'GitHub repository access for codebase analysis and file operations',
                'Real-time web access via SerpAPI for live query resolution',
                'Taipy-generated lightweight dashboard for seamless interaction',
                'Docker container orchestration through conversational interface',
                'Live agent interaction with real-time stream updates',
                'Deployment-ready containerized architecture'
            ],
            technicalDetails: {
                agentArchitecture: 'LangChain custom agent with specialized tool integration',
                dockerIntegration: 'Docker SDK for container start, stop, and management',
                githubAccess: 'GitHub REST API for repository operations and file access',
                webAccess: 'SerpAPI integration for real-time search and information retrieval',
                frontend: 'Taipy lightweight dashboard with real-time streaming capabilities',
                evaluation: 'Tool execution success rate, response accuracy, and system reliability'
            },
            results: {
                toolExecutionSuccess: '93%',
                codeAssistanceAccuracy: '91%',
                containerOrchestration: '95%',
                realTimeResponseRate: '89%'
            },
            applications: [
                'DevOps automation and container management',
                'Real-time code assistance and repository analysis',
                'AI-integrated development dashboards',
                'Full-stack development workflow automation'
            ],
            futurePlans: [
                'Add CI/CD pipeline integration capabilities',
                'Implement multi-repository project management',
                'Deploy as cloud-native development platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/devbotappai-fullstack-agent',
            demoLink: 'https://devbotappai-demo.taipy.cloud',
            tags: ['DevBotAppAI', 'LangChain Agent', 'Docker AI', 'GitHub AI', 'SerpAPI', 'Taipy App', 'Full-Stack AI', 'DevOps Automation'],
            featured: true,
            projectNumber: 91,
            totalProjects: 120,
            categoryProgress: '11/20 GenAI Projects'
        },
        {
            id: 'genai-12',
            title: 'ReportGenX: Weekly Sales Summary Generator with LangChain + Streamlit + Matplotlib',
            category: 'Generative AI',
            domain: 'Business Intelligence & Automated Reporting',
            description: 'ReportGenX is a smart weekly sales report generator that uses LangChain for LLM-driven summaries and Streamlit for an interactive UI, combined with Matplotlib for sales trend visualization.',
            image: 'port/genai/12.jpg',
            video: 'port/genai/12.mp4',
            technologies: ['Python', 'LangChain', 'Streamlit', 'Matplotlib', 'Pandas', 'PromptTemplate', 'ChatModel', 'Data Visualization'],
            frameworks: ['LangChain', 'Streamlit', 'Matplotlib', 'Pandas'],
            accuracy: '94%',
            modelSize: 'Variable (LangChain model-dependent)',
            trainingTime: 'No training (Template-based generation)',
            dataset: 'Weekly Sales Data (CSV/Excel), Business Metrics, Product Categories',
            keyFeatures: [
                'LangChain-powered intelligent sales data summarization',
                'Automated weekly performance insights with key metrics highlighting',
                'Top products and peak sales days identification',
                'Revenue growth/decline analysis with contextual explanations',
                'Interactive Matplotlib sales trend visualization',
                'Category and product-specific breakdown charts',
                'Streamlit web app with file upload and clean UI/UX',
                'Real-time report generation with visual storytelling'
            ],
            technicalDetails: {
                llmSummarization: 'LangChain PromptTemplate + ChatModel for intelligent data analysis',
                dataProcessing: 'Pandas for CSV/Excel manipulation and metrics calculation',
                visualization: 'Matplotlib for weekly trends, category breakdowns, and insights',
                interface: 'Streamlit interactive web app with file upload capabilities',
                reporting: 'Automated summary generation with visual chart integration',
                evaluation: 'Summary accuracy, insight relevance, and visualization clarity'
            },
            results: {
                summaryAccuracy: '94%',
                insightRelevance: '92%',
                visualizationClarity: '96%',
                processingSpeed: '15 seconds per report'
            },
            applications: [
                'Business analyst performance reporting',
                'Sales team weekly review automation',
                'Management dashboard and insights',
                'Automated business intelligence workflows'
            ],
            futurePlans: [
                'Add PDF export and email report distribution',
                'Implement voice summary with TTS integration',
                'Deploy as enterprise business intelligence platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/reportgenx-sales-generator',
            demoLink: 'https://reportgenx-demo.streamlit.app',
            tags: ['ReportGenX', 'Sales Insights', 'LangChain Reports', 'Streamlit App', 'AI Reporting', 'Business Analytics AI'],
            featured: true,
            projectNumber: 92,
            totalProjects: 120,
            categoryProgress: '12/20 GenAI Projects'
        },
        {
            id: 'genai-13',
            title: 'Autofixer: AI-Powered Code Debugging & Error Resolution',
            category: 'Generative AI',
            domain: 'Code Analysis & Automated Debugging',
            description: 'Autofixer is an AI-powered debugging assistant that analyzes code files and error messages to automatically generate fixed versions with detailed explanations.',
            image: 'port/genai/13.jpg',
            video: 'port/genai/13.mp4',
            technologies: ['Python', 'Streamlit', 'AI Code Analysis', 'Error Stack Trace Analysis', 'Code Generation', 'LLM Integration'],
            frameworks: ['Streamlit', 'LangChain'],
            accuracy: '91%',
            modelSize: 'Variable (LLM-dependent)',
            trainingTime: 'No training (Analysis-based)',
            dataset: 'Code Files, Error Messages, Stack Traces, Debug Patterns',
            keyFeatures: [
                'Interactive Streamlit interface for code and error upload',
                'AI-powered code analysis and error diagnosis',
                'Automatic code fixing with optimization suggestions',
                'Detailed explanations for all code changes and fixes',
                'Stack trace analysis for precise error identification',
                'Real-time debugging assistance for developers',
                'Clean UI/UX designed for both beginners and experienced developers'
            ],
            technicalDetails: {
                interface: 'Streamlit web app with file upload and text input capabilities',
                codeAnalysis: 'AI-powered source code parsing and error pattern recognition',
                errorDiagnosis: 'Stack trace analysis and error message interpretation',
                codeGeneration: 'Automated fix generation with optimization recommendations',
                explanation: 'Detailed change documentation and learning insights',
                evaluation: 'Fix accuracy, code quality improvement, and user satisfaction'
            },
            results: {
                fixAccuracy: '91%',
                codeQualityImprovement: '88%',
                debuggingSpeed: '10x faster than manual debugging',
                userSatisfaction: '94%'
            },
            applications: [
                'Beginner developer learning and assistance',
                'Rapid prototyping and debugging workflows',
                'Code review and quality assurance automation',
                'Educational programming support tools'
            ],
            futurePlans: [
                'Add multi-language support (Python, JavaScript, C++)',
                'Implement real-time debugging suggestions',
                'Deploy as IDE plugin for seamless integration'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/autofixer-ai-debugging',
            demoLink: 'https://autofixer-demo.streamlit.app',
            tags: ['Autofixer', 'AI Debugging', 'Code Analysis', 'Error Resolution', 'Developer Tools', 'Streamlit App'],
            featured: true,
            projectNumber: 93,
            totalProjects: 120,
            categoryProgress: '13/20 GenAI Projects'
        },
        {
            id: 'genai-14',
            title: 'ScheduleBot: AI-Powered Google Meeting & Calendar Manager',
            category: 'Generative AI',
            domain: 'AI Productivity & Workflow Automation',
            description: 'ScheduleBot is an AI-powered scheduling assistant that automatically manages Google Calendar, Google Meet links, and team calendars through natural language commands.',
            image: 'port/genai/14.jpg',
            video: 'port/genai/14.mp4',
            technologies: ['Python', 'Phi Data Agentic AI', 'GPT-Opus 120B', 'Google Calendar API', 'Google Sheets API', 'Flask', 'HTML/CSS', 'Docker', 'Docker Compose'],
            frameworks: ['Phi Data Agentic AI', 'Flask', 'Google APIs', 'Docker'],
            accuracy: '95%',
            modelSize: '120B (GPT-Opus)',
            trainingTime: 'Pretrained + API Integration (3 hours)',
            dataset: 'Calendar Events, Meeting Patterns, Scheduling Commands',
            keyFeatures: [
                'Direct Google Calendar and Google Sheets integration',
                'Phi Data Agentic AI Framework with GPT-Opus 120B',
                'Natural language scheduling through conversational interface',
                'Automatic Google Meet link creation and calendar invites',
                'Custom AI tool functions for workflow automation',
                'Containerized deployment with Docker and Docker Compose',
                'Web interface with Flask backend and HTML/CSS frontend',
                'Real-world workflow management through AI agents'
            ],
            technicalDetails: {
                aiFramework: 'Phi Data Agentic AI with GPT-Opus 120B for natural language processing',
                googleIntegration: 'Calendar API for event management, Sheets API for data tracking',
                webInterface: 'Flask backend with HTML/CSS frontend for smooth user experience',
                containerization: 'Docker and Docker Compose for easy deployment and scaling',
                automation: 'Custom tool functions for meeting creation and calendar management',
                evaluation: 'Scheduling accuracy, API integration success, and user workflow efficiency'
            },
            results: {
                schedulingAccuracy: '95%',
                meetingCreationSuccess: '97%',
                workflowEfficiency: '85% time savings',
                userSatisfaction: '93%'
            },
            applications: [
                'Team meeting coordination and scheduling',
                'Automated calendar management for businesses',
                'AI-powered productivity enhancement tools',
                'Enterprise workflow automation systems'
            ],
            futurePlans: [
                'Add multi-timezone scheduling intelligence',
                'Implement conflict resolution and rescheduling',
                'Deploy as enterprise SaaS productivity platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/schedulebot-ai-calendar-manager',
            demoLink: 'https://schedulebot-demo.herokuapp.com',
            tags: ['ScheduleBot', 'Phi Data Agentic AI', 'Google Calendar', 'Meeting Automation', 'AI Productivity', 'Flask App'],
            featured: true,
            projectNumber: 94,
            totalProjects: 120,
            categoryProgress: '14/20 GenAI Projects'
        },
        {
            id: 'genai-15',
            title: 'WebAutoGPT: Intelligent Web Automation with Zyte + GPT-4o',
            category: 'Generative AI',
            domain: 'Web Automation & Intelligent Data Extraction',
            description: 'WebAutoGPT is a smart automation system that combines Zyte web scraping capabilities with GPT-4o for contextual understanding and intelligent web data processing.',
            image: 'port/genai/15.jpg',
            video: 'port/genai/15.mp4',
            technologies: ['Python', 'ZyteSERPReader', 'ZyteSERPWebReader', 'GPT-4o', 'Flask', 'HTML', 'CSS', 'Web Scraping', 'Real-time Search'],
            frameworks: ['Zyte APIs', 'OpenAI GPT-4o', 'Flask'],
            accuracy: '93%',
            modelSize: 'GPT-4o (Cloud-based)',
            trainingTime: 'Pretrained + API Integration (2 hours)',
            dataset: 'Real-time Web Data, Search Results, Dynamic Content',
            keyFeatures: [
                'ZyteSERPReader for powerful real-time web search capabilities',
                'ZyteSERPWebReader for intelligent web content extraction',
                'GPT-4o integration for contextual understanding and response generation',
                'Flask web application with clean HTML/CSS frontend',
                'Real-time knowledge fetching and dynamic content processing',
                'Autonomous web agent capabilities for intelligent data gathering',
                'Seamless user experience with intuitive web interface',
                'Dynamic insight generation from live web data'
            ],
            technicalDetails: {
                webScraping: 'Zyte APIs for robust web search and content extraction',
                aiProcessing: 'GPT-4o for contextual analysis and intelligent response generation',
                webInterface: 'Flask backend with responsive HTML/CSS frontend',
                automation: 'Autonomous web data fetching and processing pipeline',
                realTimeProcessing: 'Dynamic content analysis and insight generation',
                evaluation: 'Data extraction accuracy, response relevance, and system reliability'
            },
            results: {
                dataExtractionAccuracy: '93%',
                responseRelevance: '91%',
                processingSpeed: '8 seconds per query',
                systemReliability: '96%'
            },
            applications: [
                'Autonomous web research and data gathering',
                'Real-time market intelligence and monitoring',
                'Intelligent content aggregation and analysis',
                'Automated web-based knowledge extraction'
            ],
            futurePlans: [
                'Add multi-source data correlation capabilities',
                'Implement scheduled automated web monitoring',
                'Deploy as enterprise web intelligence platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/webautogpt-zyte-automation',
            demoLink: 'https://webautogpt-demo.herokuapp.com',
            tags: ['WebAutoGPT', 'Zyte APIs', 'GPT-4o', 'Web Automation', 'Intelligent Scraping', 'Flask App', 'Real-time Data'],
            featured: true,
            projectNumber: 95,
            totalProjects: 120,
            categoryProgress: '15/20 GenAI Projects'
        },
        {
            id: 'genai-16',
            title: 'ContextArchitect: RAG-powered Knowledge Engine',
            category: 'Generative AI',
            domain: 'Retrieval-Augmented Generation & Knowledge Systems',
            description: 'ContextArchitect is a comprehensive RAG system that combines LlamaIndex orchestration, Llama 3 reasoning, and FAISS vector storage for scalable knowledge retrieval and generation.',
            image: 'port/genai/16.jpg',
            video: 'port/genai/16.mp4',
            technologies: ['Python', 'LlamaIndex', 'Llama 3', 'Hugging Face Embeddings', 'FAISS', 'Poetry', 'Streamlit', 'Custom Logging'],
            frameworks: ['LlamaIndex', 'FAISS', 'Hugging Face', 'Streamlit', 'Poetry'],
            accuracy: '94%',
            modelSize: '8B (Llama 3) + Embedding Models',
            trainingTime: 'Pretrained + Indexing (Variable based on dataset)',
            dataset: 'Custom Documents, Knowledge Bases, Text Corpora',
            keyFeatures: [
                'LlamaIndex framework for comprehensive RAG orchestration',
                'Llama 3 LLM as the core reasoning and generation engine',
                'Hugging Face embeddings for semantic document representation',
                'FAISS vector store for efficient similarity search and retrieval',
                'Poetry package management for reproducible development environment',
                'Interactive Streamlit interface for document upload and querying',
                'Custom logging system for performance monitoring and debugging',
                'Scalable, modular, and production-ready RAG architecture'
            ],
            technicalDetails: {
                orchestration: 'LlamaIndex framework for RAG pipeline management and coordination',
                reasoning: 'Llama 3 LLM for context-aware response generation',
                embeddings: 'Hugging Face models for semantic document vectorization',
                vectorStore: 'FAISS for efficient vector indexing and similarity search',
                interface: 'Streamlit web app with document upload and interactive querying',
                evaluation: 'Retrieval accuracy, response relevance, and system performance metrics'
            },
            results: {
                retrievalAccuracy: '94%',
                responseRelevance: '92%',
                querySpeed: '< 2 seconds',
                systemScalability: 'Handles 100K+ documents efficiently'
            },
            applications: [
                'Enterprise knowledge management systems',
                'Document-based question answering platforms',
                'Research and academic information retrieval',
                'Customer support knowledge bases'
            ],
            futurePlans: [
                'Add multi-modal document support (images, tables)',
                'Implement hybrid search with keyword + semantic matching',
                'Deploy as enterprise knowledge platform service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/contextarchitect-rag-engine',
            demoLink: 'https://contextarchitect-demo.streamlit.app',
            tags: ['ContextArchitect', 'RAG', 'LlamaIndex', 'Llama 3', 'FAISS', 'Vector Search', 'Knowledge Engine'],
            featured: true,
            projectNumber: 96,
            totalProjects: 120,
            categoryProgress: '16/20 GenAI Projects'
        },
        {
            id: 'genai-17',
            title: 'FinSightAI: Advanced Financial Intelligence System with RAG + Agentic AI',
            category: 'Generative AI',
            domain: 'Financial Intelligence & Multi-Agent Systems',
            description: 'FinSightAI is an advanced financial insights system combining RAG-powered query processing with Agentic AI frameworks for comprehensive financial intelligence and reporting.',
            image: 'port/genai/17.jpg',
            video: 'port/genai/17.mp4',
            technologies: ['Python', 'LlamaIndex', 'HuggingFace Embeddings', 'ChromaDB', 'Google Gemini 1.5', 'GPT-Opus-120B', 'SuperAGI', 'Gradio'],
            frameworks: ['LlamaIndex', 'ChromaDB', 'SuperAGI', 'Gradio', 'HuggingFace'],
            accuracy: '95%',
            modelSize: '120B (GPT-Opus) + Gemini 1.5 (Cloud)',
            trainingTime: 'Pretrained + Financial Data Indexing (4 hours)',
            dataset: 'Financial Reports, Market Data, Economic Indicators, Investment Analysis',
            keyFeatures: [
                'RAG-powered query system with LlamaIndex orchestration',
                'HuggingFace embeddings with ChromaDB vector storage',
                'Google Gemini 1.5 for contextual financial data processing',
                'GPT-Opus-120B enhancement for detailed financial insights',
                'SuperAGI integration for conversational AI report generation',
                'Interactive Gradio interface for real-time financial intelligence',
                'Multi-agent orchestration for comprehensive financial analysis',
                'Dynamic structured financial report generation'
            ],
            technicalDetails: {
                ragSystem: 'LlamaIndex + HuggingFace embeddings + ChromaDB for financial data retrieval',
                aiProcessing: 'Google Gemini 1.5 for contextual accuracy, GPT-Opus-120B for enhancement',
                agenticFramework: 'SuperAGI for conversational AI and dynamic report generation',
                interface: 'Gradio interactive web interface for seamless user experience',
                dataProcessing: 'Multi-modal financial data analysis and insight generation',
                evaluation: 'Financial accuracy, insight relevance, and report quality metrics'
            },
            results: {
                financialAccuracy: '95%',
                insightRelevance: '93%',
                reportQuality: '94%',
                queryResponseTime: '3-5 seconds'
            },
            applications: [
                'Investment analysis and portfolio management',
                'Financial research and market intelligence',
                'Automated financial reporting systems',
                'Enterprise financial decision support tools'
            ],
            futurePlans: [
                'Add real-time market data integration',
                'Implement predictive financial modeling',
                'Deploy as enterprise financial intelligence platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/finsightai-financial-intelligence',
            demoLink: 'https://finsightai-demo.gradio.app',
            tags: ['FinSightAI', 'Financial AI', 'RAG', 'SuperAGI', 'ChromaDB', 'Gemini 1.5', 'GPT-Opus', 'Financial Intelligence'],
            featured: true,
            projectNumber: 97,
            totalProjects: 120,
            categoryProgress: '17/20 GenAI Projects'
        },
        {
            id: 'genai-18',
            title: 'MedicoSage: Medical Knowledge Assistant with RAG + LLMs',
            category: 'Generative AI',
            domain: 'Healthcare AI & Medical Knowledge Systems',
            description: 'MedicoSage is a revolutionary medical knowledge assistance system designed to support healthcare professionals and researchers with RAG-powered medical literature retrieval and LLM reasoning.',
            image: 'port/genai/18.jpg',
            video: 'port/genai/18.mp4',
            technologies: ['Python', 'LlamaIndex', 'Pinecone', 'GPT-Opus-120B', 'Streamlit', 'Medical Literature Processing', 'Vector Search'],
            frameworks: ['LlamaIndex', 'Pinecone', 'Streamlit', 'OpenAI'],
            accuracy: '96%',
            modelSize: '120B (GPT-Opus)',
            trainingTime: 'Pretrained + Medical Data Indexing (6 hours)',
            dataset: 'Medical Literature, Research Papers, Clinical Guidelines, Healthcare Datasets',
            keyFeatures: [
                'Medical literature and dataset indexing using LlamaIndex',
                'Pinecone vector database for lightning-fast semantic search',
                'GPT-Opus-120B for context-aware, evidence-based medical reasoning',
                'Interactive Streamlit interface for healthcare professionals',
                'Real-time medical query resolution and knowledge retrieval',
                'Evidence-based answer generation with source citations',
                'Healthcare-specific RAG optimization for medical accuracy',
                'Support for medical research and clinical decision-making'
            ],
            technicalDetails: {
                knowledgeRetrieval: 'LlamaIndex medical literature indexing with Pinecone vector storage',
                reasoning: 'GPT-Opus-120B for context-aware medical knowledge synthesis',
                interface: 'Streamlit web application optimized for healthcare workflows',
                vectorSearch: 'Pinecone semantic search for rapid medical literature retrieval',
                medicalAccuracy: 'Healthcare-specific prompt engineering and validation',
                evaluation: 'Medical accuracy, evidence quality, and clinical relevance metrics'
            },
            results: {
                medicalAccuracy: '96%',
                evidenceQuality: '94%',
                querySpeed: '< 2 seconds',
                clinicalRelevance: '95%'
            },
            applications: [
                'Clinical decision support systems',
                'Medical research assistance and literature review',
                'Healthcare professional education and training',
                'Evidence-based medicine and treatment planning'
            ],
            futurePlans: [
                'Add medical imaging analysis capabilities',
                'Implement clinical trial data integration',
                'Deploy as enterprise healthcare knowledge platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/medicosage-medical-ai',
            demoLink: 'https://medicosage-demo.streamlit.app',
            tags: ['MedicoSage', 'Healthcare AI', 'Medical Knowledge', 'RAG', 'Pinecone', 'GPT-Opus', 'Clinical Support'],
            featured: true,
            projectNumber: 98,
            totalProjects: 120,
            categoryProgress: '18/20 GenAI Projects'
        },
        {
            id: 'genai-19',
            title: 'AI-Powered MCQ Generator for Education & Assessments',
            category: 'Generative AI',
            domain: 'Educational Technology & Assessment Automation',
            description: 'AI-Powered MCQ Generator is an intelligent assessment creation system that uses LangChain and GPT-Opus-120B to automatically generate multiple choice questions from any text or dataset.',
            image: 'port/genai/19.jpg',
            video: 'port/genai/19.mp4',
            technologies: ['Python', 'LangChain', 'GPT-Opus-120B', 'Flask API', 'HTML', 'CSS', 'JavaScript', 'Educational AI'],
            frameworks: ['LangChain', 'Flask', 'OpenAI'],
            accuracy: '93%',
            modelSize: '120B (GPT-Opus)',
            trainingTime: 'Pretrained + Educational Prompt Engineering (2 hours)',
            dataset: 'Educational Content, Textbooks, Academic Materials, Assessment Data',
            keyFeatures: [
                'Automated MCQ generation from any text or dataset using LangChain',
                'GPT-Opus-120B integration for high-quality question formulation',
                'API-first design with Flask backend for platform integration',
                'User-friendly web interface built with HTML, CSS, and JavaScript',
                'Dynamic question creation with customizable difficulty levels',
                'Educational content analysis and question optimization',
                'Seamless integration capabilities for EdTech platforms',
                'Adaptive assessment generation for personalized learning'
            ],
            technicalDetails: {
                questionGeneration: 'LangChain orchestration with GPT-Opus-120B for intelligent MCQ creation',
                apiBackend: 'Flask RESTful API for seamless platform integration',
                frontend: 'Responsive web interface with HTML, CSS, and JavaScript',
                contentAnalysis: 'Automated text processing and educational content extraction',
                customization: 'Configurable difficulty levels and question types',
                evaluation: 'Question quality, educational relevance, and generation accuracy'
            },
            results: {
                questionQuality: '93%',
                educationalRelevance: '91%',
                generationSpeed: '10 questions per minute',
                apiReliability: '97%'
            },
            applications: [
                'Educational assessment automation',
                'Online learning platform integration',
                'Teacher productivity enhancement tools',
                'Adaptive learning and personalized education'
            ],
            futurePlans: [
                'Add support for different question types (fill-in-blank, true/false)',
                'Implement difficulty progression algorithms',
                'Deploy as SaaS EdTech platform service'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/ai-mcq-generator-langchain',
            demoLink: 'https://ai-mcq-generator-demo.herokuapp.com',
            tags: ['MCQ Generator', 'EdTech', 'LangChain', 'GPT-Opus', 'Educational AI', 'Assessment Automation', 'Flask API'],
            featured: true,
            projectNumber: 99,
            totalProjects: 120,
            categoryProgress: '19/20 GenAI Projects'
        },
        {
            id: 'genai-20',
            title: 'LLM Studio Pro: Unified Chat, Agents & RAG Platform',
            category: 'Generative AI',
            domain: 'Comprehensive LLM Development Platform',
            description: 'LLM Studio Pro is an advanced unified platform that brings together Chat, Agents, and RAG pipelines under one comprehensive system for LLM experimentation and development.',
            image: 'port/genai/20.jpg',
            video: 'port/genai/20.mp4',
            technologies: ['Python', 'Streamlit', 'DeepSeek', 'OpenAI', 'LLaMA', 'LangChain Components', 'Vector Stores', 'Tool Integration'],
            frameworks: ['Streamlit', 'LangChain', 'Multiple LLM APIs'],
            accuracy: '96%',
            modelSize: 'Multi-LLM (DeepSeek, OpenAI, LLaMA)',
            trainingTime: 'Pretrained + Platform Integration (8 hours)',
            dataset: 'Multi-domain Knowledge Base, Tool Definitions, Conversation Patterns',
            keyFeatures: [
                'Unified platform combining Chat, Agents, and RAG capabilities',
                'Multi-LLM integration (DeepSeek, OpenAI, LLaMA) for diverse conversations',
                'Advanced Agent Module with real-time tool execution',
                'Comprehensive RAG Module with flexible component architecture',
                'Modular dataloaders, splitters, embeddings, and vector stores',
                'Interactive Streamlit interface for seamless user experience',
                'Comprehensive playground for AI experimentation and research',
                'Tool-augmented agents for complex workflow handling'
            ],
            technicalDetails: {
                chatModule: 'Multi-LLM integration with seamless conversation switching',
                agentModule: 'Real-time tool execution and complex workflow management',
                ragModule: 'Flexible RAG components with customizable retrieval pipelines',
                platform: 'Streamlit-based unified interface for all AI capabilities',
                toolIntegration: 'Extensible tool system for agent augmentation',
                evaluation: 'Cross-module performance, user experience, and system reliability'
            },
            results: {
                platformReliability: '96%',
                multiLLMPerformance: '94%',
                toolExecutionSuccess: '92%',
                userExperience: '95%'
            },
            applications: [
                'AI research and development playground',
                'LLM experimentation and prototyping platform',
                'Educational AI learning environment',
                'Enterprise AI solution development'
            ],
            futurePlans: [
                'Add collaborative features for team development',
                'Implement custom model fine-tuning capabilities',
                'Deploy as cloud-native AI development platform'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/llm-studio-pro-platform',
            demoLink: 'https://llm-studio-pro-demo.streamlit.app',
            tags: ['LLM Studio Pro', 'Multi-LLM Platform', 'Chat Agents', 'RAG System', 'AI Playground', 'Unified Platform'],
            featured: true,
            projectNumber: 100,
            totalProjects: 120,
            categoryProgress: '20/20 GenAI Projects - COMPLETED! 🎉'
        }
    ],

    // 📊 Data Analytics Projects
    analytics: [
        {
            id: 'analytics-1',
            title: 'Customer Segmentation using K-Means Clustering',
            category: 'Data Analytics',
            domain: 'Business Intelligence & Customer Analytics',
            description: 'Applied K-Means clustering to segment customers based on purchase behavior, uncovering distinct customer groups for targeted marketing strategies and business insights.',
            image: 'port/analytics/1.jpg',
            video: 'port/analytics/1.mp4',
            technologies: ['Python', 'Pandas', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'K-Means', 'Data Visualization'],
            frameworks: ['Scikit-learn', 'Matplotlib', 'Seaborn'],
            accuracy: '85%',
            modelSize: '5MB',
            trainingTime: '30 minutes',
            dataset: 'Customer Purchase Behavior Dataset',
            keyFeatures: [
                'K-Means clustering for customer behavior segmentation',
                'Purchase pattern analysis and customer profiling',
                'Distinct customer group identification (high-value, occasional, budget-focused)',
                'Comprehensive data visualization with cluster analysis',
                'Actionable business insights for marketing strategy optimization',
                'Statistical analysis of customer segments and characteristics',
                'Interactive visualizations for business stakeholder communication'
            ],
            technicalDetails: {
                clustering: 'K-Means algorithm with optimal cluster determination',
                analysis: 'Purchase behavior pattern recognition and segmentation',
                visualization: 'Matplotlib and Seaborn for cluster visualization and insights',
                businessIntelligence: 'Customer profiling and marketing strategy recommendations',
                dataProcessing: 'Pandas for data manipulation and feature engineering',
                evaluation: 'Silhouette score, inertia, and business relevance metrics'
            },
            results: {
                segmentationAccuracy: '85%',
                customerGroups: '4 distinct segments identified',
                businessInsights: '92% actionable recommendations',
                visualizationClarity: '94%'
            },
            applications: [
                'Personalized marketing campaign development',
                'Customer loyalty program optimization',
                'Targeted product recommendations',
                'Business strategy and customer retention planning'
            ],
            futurePlans: [
                'Implement hierarchical clustering for comparison',
                'Add real-time customer segmentation capabilities',
                'Deploy as business intelligence dashboard'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/customer-segmentation-kmeans-analytics',
            demoLink: 'https://customer-segmentation-analytics-demo.streamlit.app',
            tags: ['Customer Segmentation', 'K-Means', 'Business Intelligence', 'Customer Analytics', 'Data Visualization', 'Marketing Strategy'],
            featured: true,
            projectNumber: 101,
            totalProjects: 120,
            categoryProgress: '1/20 Analytics Projects'
        },
        {
            id: 'analytics-2',
            title: 'Interactive Sales Dashboard using Excel',
            category: 'Data Analytics',
            domain: 'Business Intelligence & Sales Analytics',
            description: 'Built an interactive Sales Dashboard in Excel to analyze company-wide sales performance with dynamic filtering and comprehensive KPI tracking for data-driven business decisions.',
            image: 'port/analytics/2.jpg',
            video: 'port/analytics/2.mp4',
            technologies: ['Microsoft Excel', 'Pivot Tables', 'Charts', 'Slicers', 'Conditional Formatting', 'Data Visualization'],
            frameworks: ['Microsoft Excel'],
            accuracy: '100%',
            modelSize: 'N/A (Excel-based)',
            trainingTime: 'No training (Dashboard creation: 4 hours)',
            dataset: 'Company Sales Data (Multi-year, Multi-region)',
            keyFeatures: [
                'Comprehensive sales performance analysis with KPI tracking',
                'Total Profit: ₹82,26,755.66 and Total Sales: ₹1,00,32,628.85',
                'Maximum single deal profit identification: ₹11,547.90',
                'Top performing city analysis (Madrid as highest profit generator)',
                'Multi-dimensional breakdown by Product Line, Deal Size, Year, and Country',
                'Dynamic slicers for interactive filtering (city, year, country)',
                'Professional dashboard design with conditional formatting'
            ],
            technicalDetails: {
                dataProcessing: 'Excel Pivot Tables for data aggregation and analysis',
                visualization: 'Charts, graphs, and conditional formatting for insights',
                interactivity: 'Slicers for dynamic filtering and real-time updates',
                kpiTracking: 'Key performance indicators with visual representations',
                businessIntelligence: 'Multi-dimensional sales analysis and trend identification',
                evaluation: 'Data accuracy, visualization clarity, and business insight generation'
            },
            results: {
                totalProfit: '₹82,26,755.66',
                totalSales: '₹1,00,32,628.85',
                maxSingleProfit: '₹11,547.90',
                topCity: 'Madrid',
                dashboardEfficiency: '95% user satisfaction'
            },
            applications: [
                'Executive sales performance reporting',
                'Regional sales team management',
                'Product line profitability analysis',
                'Strategic business planning and forecasting'
            ],
            futurePlans: [
                'Migrate to Power BI for advanced analytics',
                'Add predictive forecasting capabilities',
                'Implement real-time data connections'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/excel-sales-dashboard-analytics',
            demoLink: 'https://excel-sales-dashboard-demo.onedrive.live.com',
            tags: ['Excel Dashboard', 'Sales Analytics', 'Business Intelligence', 'KPI Tracking', 'Data Visualization', 'Interactive Reports'],
            featured: true,
            projectNumber: 102,
            totalProjects: 120,
            categoryProgress: '2/20 Analytics Projects'
        },
        {
            id: 'analytics-3',
            title: 'Product Performance Dashboard using Power BI',
            category: 'Data Analytics',
            domain: 'Business Intelligence & Product Analytics',
            description: 'Built a comprehensive Product Performance Dashboard using Power BI to analyze product sales and revenue performance from 2010-2024, enabling data-driven business decisions.',
            image: 'port/analytics/3.jpg',
            video: 'port/analytics/3.mp4',
            technologies: ['Power BI', 'Power Query', 'DAX', 'Data Modeling', 'Interactive Visualizations', 'KPI Analysis'],
            frameworks: ['Microsoft Power BI'],
            accuracy: '98%',
            modelSize: 'N/A (BI Dashboard)',
            trainingTime: 'No training (Dashboard development: 6 hours)',
            dataset: 'Product Sales & Revenue Data (2010-2024)',
            keyFeatures: [
                'Comprehensive product performance analysis with 14-year data span',
                'Key Performance Indicators: Max Product Sale (19M), Max Revenue (77.94M), Overall Revenue (236.13M)',
                'Interactive visualizations: Line charts, stacked bars, and performance gauges',
                'Product leadership identification: P1 (sales leader), P3 (revenue leader)',
                'Dynamic filtering capabilities for deep-dive analysis',
                'Time-series trend analysis with peak performance identification (2012-2016)',
                'Underperformer identification and growth opportunity analysis'
            ],
            technicalDetails: {
                dataProcessing: 'Power Query for data transformation and cleaning',
                calculations: 'DAX formulas for advanced KPI calculations and measures',
                visualization: 'Interactive charts, gauges, and dynamic filtering',
                modeling: 'Optimized data model for performance and scalability',
                businessIntelligence: 'Actionable insights generation and trend analysis',
                evaluation: 'Dashboard performance, user engagement, and business impact metrics'
            },
            results: {
                maxProductSale: '19M units',
                maxRevenue: '77.94M',
                overallRevenue: '236.13M',
                topSalesProduct: 'Product P1',
                topRevenueProduct: 'Product P3',
                peakPeriod: '2012-2016'
            },
            applications: [
                'Product portfolio optimization',
                'Sales strategy development',
                'Revenue forecasting and planning',
                'Executive performance reporting'
            ],
            futurePlans: [
                'Add predictive analytics for future performance',
                'Implement real-time data refresh capabilities',
                'Deploy advanced AI-powered insights'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/powerbi-product-performance-dashboard',
            demoLink: 'https://app.powerbi.com/view?r=product-performance-demo',
            tags: ['Power BI', 'Product Analytics', 'Business Intelligence', 'KPI Dashboard', 'Revenue Analysis', 'Performance Tracking'],
            featured: true,
            projectNumber: 103,
            totalProjects: 120,
            categoryProgress: '3/20 Analytics Projects'
        },
        {
            id: 'analytics-4',
            title: 'Retail Sales Analysis using Python',
            category: 'Data Analytics',
            domain: 'Retail Analytics & Exploratory Data Analysis',
            description: 'Comprehensive retail sales analysis using Python to uncover patterns and trends in transactional data, providing actionable insights for business optimization and strategic decision-making.',
            image: 'port/analytics/4.jpg',
            video: 'port/analytics/4.mp4',
            technologies: ['Python', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Exploratory Data Analysis', 'Data Visualization'],
            frameworks: ['Pandas', 'Matplotlib', 'Seaborn'],
            accuracy: '96%',
            modelSize: 'N/A (Analysis-based)',
            trainingTime: 'No training (Analysis: 5 hours)',
            dataset: 'Retail Transactional Sales Data (Multi-category, Multi-region)',
            keyFeatures: [
                'Comprehensive retail sales data processing and cleaning with Python',
                'Exploratory Data Analysis (EDA) for pattern identification',
                'Monthly and seasonal trend analysis for sales optimization',
                'Top-performing product category identification and revenue analysis',
                'Customer purchase behavior analysis and segmentation',
                'Peak sales months and regional performance identification',
                'Advanced data visualization with charts and statistical plots'
            ],
            technicalDetails: {
                dataProcessing: 'Pandas for data cleaning, transformation, and manipulation',
                analysis: 'NumPy for statistical calculations and trend analysis',
                visualization: 'Matplotlib and Seaborn for comprehensive data visualization',
                eda: 'Exploratory data analysis with correlation, distribution, and trend analysis',
                insights: 'Business intelligence generation through statistical analysis',
                evaluation: 'Data quality assessment, insight accuracy, and business relevance'
            },
            results: {
                dataQuality: '96% after cleaning and processing',
                insightAccuracy: '94% business relevance',
                peakSalesIdentified: 'Seasonal patterns and top months discovered',
                topCategories: 'Revenue-driving product categories identified',
                customerSegments: 'High-profit customer groups analyzed'
            },
            applications: [
                'Inventory planning and optimization',
                'Marketing strategy development',
                'Pricing decision support',
                'Customer targeting and segmentation'
            ],
            futurePlans: [
                'Implement predictive sales forecasting models',
                'Add real-time analytics capabilities',
                'Deploy interactive dashboard for stakeholders'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/retail-sales-analysis-python',
            demoLink: 'https://retail-sales-analysis-demo.streamlit.app',
            tags: ['Retail Analytics', 'Python Analysis', 'EDA', 'Sales Patterns', 'Business Intelligence', 'Data Visualization'],
            featured: true,
            projectNumber: 104,
            totalProjects: 120,
            categoryProgress: '4/20 Analytics Projects'
        },
        {
            id: 'analytics-5',
            title: 'Movie Rating Analysis using Tableau',
            category: 'Data Analytics',
            domain: 'Entertainment Analytics & Visual Intelligence',
            description: 'Created an interactive Movie Ratings Dashboard using Tableau to analyze audience voting patterns, film performance, and entertainment industry insights through comprehensive data visualization.',
            image: 'port/analytics/5.jpg',
            video: 'port/analytics/5.mp4',
            technologies: ['Tableau', 'Excel', 'Data Cleaning', 'Visual Analytics', 'KPI Analysis', 'Interactive Dashboards'],
            frameworks: ['Tableau Desktop'],
            accuracy: '97%',
            modelSize: 'N/A (Dashboard-based)',
            trainingTime: 'No training (Dashboard creation: 4 hours)',
            dataset: 'Movie Ratings and Voting Data (504 films)',
            keyFeatures: [
                'Interactive movie ratings dashboard with comprehensive KPI tracking',
                'Key Performance Indicators: Total Films (504), Average Rating (3.37), Highest Voted Film (34,846 votes)',
                'Multi-dimensional visualizations: Votes by Rating Distribution, Rating vs Sum of Votes',
                'Top Films analysis by Total Votes with popularity rankings',
                'Audience sentiment analysis through rating distribution patterns',
                'Film engagement correlation analysis between ratings and vote counts',
                'Entertainment industry insights for studios and analysts'
            ],
            technicalDetails: {
                dataPreparation: 'Excel-based data cleaning and preprocessing',
                visualization: 'Tableau interactive charts, KPIs, and dashboard design',
                analytics: 'Statistical analysis of rating distributions and voting patterns',
                interactivity: 'Dynamic filtering and drill-down capabilities',
                insights: 'Audience behavior analysis and film popularity correlation',
                evaluation: 'Dashboard usability, insight accuracy, and business relevance'
            },
            results: {
                totalFilms: '504 movies analyzed',
                averageRating: '3.37 overall rating',
                highestVotes: '34,846 votes (top film)',
                ratingRange: 'Most ratings between 3.0-4.5',
                topFilms: 'Fifty Shades of Grey, Jurassic World, American Sniper'
            },
            applications: [
                'Film studio performance analysis',
                'Audience engagement measurement',
                'Entertainment industry market research',
                'Content strategy and investment decisions'
            ],
            futurePlans: [
                'Add predictive analytics for box office success',
                'Implement real-time streaming data integration',
                'Deploy advanced sentiment analysis capabilities'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/movie-rating-analysis-tableau',
            demoLink: 'https://public.tableau.com/views/MovieRatingAnalysis/Dashboard',
            tags: ['Tableau Dashboard', 'Movie Analytics', 'Entertainment Data', 'Visual Analytics', 'Audience Insights', 'KPI Tracking'],
            featured: true,
            projectNumber: 105,
            totalProjects: 120,
            categoryProgress: '5/20 Analytics Projects'
        },
        {
            id: 'analytics-6',
            title: 'Employee Performance Dashboard using Power BI',
            category: 'Data Analytics',
            domain: 'HR Analytics & Workforce Intelligence',
            description: 'Designed a comprehensive Employee Performance Dashboard using Power BI to visualize workforce metrics, performance analytics, and organizational insights across departments and roles.',
            image: 'port/analytics/6.jpg',
            video: 'port/analytics/6.mp4',
            technologies: ['Power BI', 'Power Query', 'DAX', 'Data Modeling', 'HR Analytics', 'Interactive Visualizations'],
            frameworks: ['Microsoft Power BI'],
            accuracy: '98%',
            modelSize: 'N/A (BI Dashboard)',
            trainingTime: 'No training (Dashboard development: 5 hours)',
            dataset: 'Employee Performance Data (100K total employees)',
            keyFeatures: [
                'Comprehensive workforce analytics with 100K total employees and 90K current employees',
                'Multi-dimensional analysis by Gender, Job Title, Department, and Performance Score',
                'Interactive filtering capabilities by Hire Year and Hire Month for temporal analysis',
                'Professional visualization suite with bar charts and pie charts for clarity',
                'Performance score distribution analysis showing workforce efficiency (4-5 range)',
                'Departmental breakdown highlighting IT, Sales, and Engineering as top departments',
                'Hiring trend analysis from 2014-2024 showing consistent organizational growth'
            ],
            technicalDetails: {
                dataProcessing: 'Power Query for HR data transformation and cleaning',
                calculations: 'DAX formulas for performance metrics and KPI calculations',
                modeling: 'Optimized data model for workforce analytics and reporting',
                visualization: 'Interactive charts with dynamic filtering and drill-down capabilities',
                hrAnalytics: 'Employee distribution, performance tracking, and talent insights',
                evaluation: 'Dashboard performance, user engagement, and HR decision support metrics'
            },
            results: {
                totalEmployees: '100K employees analyzed',
                currentEmployees: '90K active workforce',
                performanceRange: '4-5 score range (majority)',
                topDepartments: 'IT, Sales, Engineering',
                hiringTrend: 'Consistent growth 2014-2024',
                genderBalance: 'Balanced representation achieved'
            },
            applications: [
                'HR strategic planning and workforce optimization',
                'Performance management and talent development',
                'Organizational structure analysis and planning',
                'Data-driven recruitment and retention strategies'
            ],
            futurePlans: [
                'Add predictive analytics for employee retention',
                'Implement real-time performance tracking',
                'Deploy advanced talent analytics and succession planning'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/employee-performance-powerbi-dashboard',
            demoLink: 'https://app.powerbi.com/view?r=employee-performance-demo',
            tags: ['Power BI', 'HR Analytics', 'Employee Performance', 'Workforce Intelligence', 'Organizational Analytics', 'Talent Management'],
            featured: true,
            projectNumber: 106,
            totalProjects: 120,
            categoryProgress: '6/20 Analytics Projects'
        },
        {
            id: 'analytics-7',
            title: 'Customer Churn Classification using Logistic Regression',
            category: 'Data Analytics',
            domain: 'Customer Analytics & Predictive Modeling',
            description: 'Comprehensive customer churn analysis using Logistic Regression to identify patterns and understand customer retention behavior, transforming raw data into actionable business insights.',
            image: 'port/analytics/7.jpg',
            video: 'port/analytics/7.mp4',
            technologies: ['Python', 'Pandas', 'NumPy', 'Scikit-learn', 'Logistic Regression', 'Matplotlib', 'Seaborn', 'EDA'],
            frameworks: ['Scikit-learn', 'Matplotlib', 'Seaborn'],
            accuracy: '85%',
            modelSize: '2MB',
            trainingTime: '15 minutes',
            dataset: 'Customer Behavior and Churn Dataset',
            keyFeatures: [
                'Comprehensive customer churn analysis with predictive classification',
                'Advanced data cleaning, feature encoding, and exploratory data analysis (EDA)',
                'Logistic Regression implementation for binary churn classification',
                'Multi-factor analysis: tenure, contract type, and monthly charges impact',
                'Correlation analysis and churn distribution visualization',
                'Customer retention strategy insights and recommendations',
                'Statistical analysis of churn patterns and business drivers'
            ],
            technicalDetails: {
                dataProcessing: 'Pandas for data cleaning, feature encoding, and transformation',
                modeling: 'Scikit-learn Logistic Regression for binary classification',
                analysis: 'NumPy for statistical calculations and pattern analysis',
                visualization: 'Matplotlib and Seaborn for correlation and distribution plots',
                eda: 'Comprehensive exploratory data analysis for churn trend identification',
                evaluation: 'Classification accuracy, precision, recall, and business impact metrics'
            },
            results: {
                classificationAccuracy: '85%',
                churnPrediction: 'Short-term monthly contract customers highest risk',
                keyFactors: 'High monthly charges and lack of long-term plans',
                businessInsights: 'Clear retention strategy recommendations generated',
                modelPerformance: 'Strong predictive capability for customer behavior'
            },
            applications: [
                'Customer retention strategy development',
                'Proactive churn prevention campaigns',
                'Pricing strategy optimization',
                'Customer lifetime value enhancement'
            ],
            futurePlans: [
                'Implement ensemble methods for improved accuracy',
                'Add real-time churn prediction capabilities',
                'Deploy automated retention intervention system'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/customer-churn-logistic-regression',
            demoLink: 'https://customer-churn-analysis-demo.streamlit.app',
            tags: ['Customer Churn', 'Logistic Regression', 'Predictive Analytics', 'Customer Retention', 'Business Intelligence', 'EDA'],
            featured: true,
            projectNumber: 107,
            totalProjects: 120,
            categoryProgress: '7/20 Analytics Projects'
        },
        {
            id: 'analytics-8',
            title: 'HR Attrition Dashboard using Power BI',
            category: 'Data Analytics',
            domain: 'HR Analytics & Employee Retention Intelligence',
            description: 'Created a comprehensive HR Attrition Dashboard using Power BI to analyze employee retention trends, attrition patterns, and workforce dynamics for data-driven HR decision making.',
            image: 'port/analytics/8.jpg',
            video: 'port/analytics/8.mp4',
            technologies: ['Power BI', 'Power Query', 'DAX', 'Data Modeling', 'HR Analytics', 'Workforce Intelligence'],
            frameworks: ['Microsoft Power BI'],
            accuracy: '97%',
            modelSize: 'N/A (BI Dashboard)',
            trainingTime: 'No training (Dashboard development: 6 hours)',
            dataset: 'Employee Attrition and HR Data (1,470 total employees)',
            keyFeatures: [
                'Comprehensive workforce analytics with 1,470 total employees and 1,233 current employees',
                'Attrition rate analysis: 16% employee turnover vs 84% retention rate',
                'Demographic insights: 60% Male, 40% Female distribution with maximum income ₹19,999',
                'Multi-dimensional analysis by Education Level, Job Satisfaction, and Department',
                'Job satisfaction correlation with retention (levels 3 & 4 showing high satisfaction)',
                'Departmental attrition comparison highlighting Sales and R&D higher turnover',
                'Travel frequency impact analysis revealing burnout correlation with attrition'
            ],
            technicalDetails: {
                dataProcessing: 'Power Query for HR data transformation and attrition calculations',
                calculations: 'DAX formulas for retention rates, attrition metrics, and KPIs',
                modeling: 'Optimized data model for workforce analytics and trend analysis',
                visualization: 'Interactive dashboards with demographic and departmental breakdowns',
                hrIntelligence: 'Employee retention insights, satisfaction analysis, and risk identification',
                evaluation: 'Dashboard effectiveness, HR decision support, and retention strategy impact'
            },
            results: {
                totalEmployees: '1,470 workforce analyzed',
                currentEmployees: '1,233 active employees',
                attritionRate: '16% turnover, 84% retention',
                genderDistribution: '60% Male, 40% Female',
                maxIncome: '₹19,999',
                highRiskDepartments: 'Sales and R&D departments'
            },
            applications: [
                'Employee retention strategy development',
                'HR risk assessment and intervention planning',
                'Workforce optimization and talent management',
                'Organizational culture and satisfaction improvement'
            ],
            futurePlans: [
                'Add predictive attrition modeling capabilities',
                'Implement real-time employee sentiment tracking',
                'Deploy advanced retention intervention recommendations'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/hr-attrition-powerbi-dashboard',
            demoLink: 'https://app.powerbi.com/view?r=hr-attrition-demo',
            tags: ['Power BI', 'HR Analytics', 'Employee Attrition', 'Workforce Intelligence', 'Retention Analysis', 'Organizational Analytics'],
            featured: true,
            projectNumber: 108,
            totalProjects: 120,
            categoryProgress: '8/20 Analytics Projects'
        },
        {
            id: 'analytics-9',
            title: 'Geospatial Analysis of Coffee Sales (2023) using Python',
            category: 'Data Analytics',
            domain: 'Geospatial Analytics & Retail Intelligence',
            description: 'Comprehensive geospatial and sales analysis of Coffee Sales Data (2023) using Python to explore location-based trends, product performance, and regional business insights.',
            image: 'port/analytics/9.jpg',
            video: 'port/analytics/9.mp4',
            technologies: ['Python', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly', 'Geopandas', 'Geospatial Analysis'],
            frameworks: ['Pandas', 'Plotly', 'Geopandas', 'Matplotlib'],
            accuracy: '96%',
            modelSize: 'N/A (Analysis-based)',
            trainingTime: 'No training (Analysis: 6 hours)',
            dataset: 'Coffee Sales Data 2023 (Multi-location, Multi-product)',
            keyFeatures: [
                'Comprehensive coffee sales analysis with 1,49,116 total transactions',
                'Geospatial intelligence across multiple store locations with 2,14,470 quantity sold',
                'Month-over-month growth analysis showing consistent sales and transaction increases',
                'Product category performance analysis with Coffee and Tea leading sales volume',
                'Regional performance comparison revealing balanced distribution across locations',
                'Interactive geospatial visualizations using Plotly and Geopandas',
                'Business intelligence insights for scaling and marketing strategies'
            ],
            technicalDetails: {
                dataProcessing: 'Pandas for sales data manipulation and transaction analysis',
                geospatialAnalysis: 'Geopandas for location-based analysis and mapping',
                visualization: 'Plotly for interactive maps, Matplotlib/Seaborn for statistical plots',
                analytics: 'NumPy for statistical calculations and trend analysis',
                insights: 'Regional performance comparison and product category analysis',
                evaluation: 'Sales accuracy, geographic distribution analysis, and business impact metrics'
            },
            results: {
                totalTransactions: '1,49,116 transactions analyzed',
                totalQuantity: '2,14,470 units sold',
                growthTrend: 'Consistent month-over-month growth',
                topCategories: 'Coffee and Tea leading sales volume',
                regionalBalance: 'Equal performance across all three store locations'
            },
            applications: [
                'Retail location optimization and expansion planning',
                'Product portfolio management and inventory optimization',
                'Regional marketing strategy development',
                'Business scaling and franchise location analysis'
            ],
            futurePlans: [
                'Add predictive analytics for location-based demand forecasting',
                'Implement real-time geospatial sales monitoring',
                'Deploy interactive business intelligence dashboard'
            ],
            githubLink: 'https://github.com/Rajkumar-Ai-Engineer/geospatial-coffee-sales-analysis',
            demoLink: 'https://coffee-sales-geospatial-demo.streamlit.app',
            tags: ['Geospatial Analysis', 'Coffee Sales', 'Python Analytics', 'Retail Intelligence', 'Location Analytics', 'Business Intelligence'],
            featured: true,
            projectNumber: 109,
            totalProjects: 120,
            categoryProgress: '9/20 Analytics Projects'
        }
    ]
};

// Function to populate projects in the portfolio
function populateProjects() {
    const allProjects = [];
    
    // Combine all projects from different categories
    Object.keys(projectDatabase).forEach(category => {
        projectDatabase[category].forEach(project => {
            allProjects.push({
                ...project,
                category: category,
                id: Date.now() + Math.random()
            });
        });
    });
    
    // Store in localStorage for the portfolio system
    localStorage.setItem('portfolioProjects', JSON.stringify(allProjects));
    
    return allProjects;
}

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { projectDatabase, populateProjects };
}
    110: {
        title: "Social Media Sentiment Analysis using Reddit Data",
        category: "analytics",
        description: "Understanding public opinions and emotions on social media by performing Sentiment Analysis on Reddit posts using Python and MongoDB with real-time data processing and visualization.",
        technologies: ["Python", "Requests", "MongoDB", "TextBlob", "Matplotlib", "Pandas"],
        frameworks: ["TextBlob", "MongoDB", "Matplotlib"],
        accuracy: "Positive sentiment majority detected",
        dataset: "Reddit posts via API",
        keyFeatures: [
            "Reddit data extraction using requests library",
            "Scalable MongoDB storage for social media data",
            "TextBlob sentiment classification (Positive/Negative/Neutral)",
            "Real-time sentiment analysis pipeline",
            "Interactive sentiment distribution visualization",
            "Community engagement pattern analysis"
        ],
        results: {
            "dataProcessed": "Real-time Reddit posts",
            "sentimentAccuracy": "High precision classification",
            "insights": "Majority positive sentiment with valuable negative/neutral trend analysis",
            "visualization": "Sentiment vs Number of Posts bar chart"
        },
        applications: [
            "Social media monitoring",
            "Brand sentiment tracking",
            "Public opinion analysis",
            "Community engagement insights",
            "Real-time sentiment dashboards"
        ],
        futurePlans: [
            "Multi-platform sentiment analysis",
            "Advanced NLP models integration",
            "Real-time sentiment alerts",
            "Emotion detection beyond sentiment"
        ],
        github: "https://github.com/username/reddit-sentiment-analysis",
        demo: "https://reddit-sentiment-demo.herokuapp.com",
        image: "port/analytics/10.jpg"
    },

    // Project 111 will be added here
    // Analytics Project 11/20 - E-Commerce Sales Analysis using SQL
    {
        id: 111,
        title: "E-Commerce Sales Analysis using SQL",
        category: "analytics",
        image: "port/analytics/11.jpg",
        technologies: ["MySQL", "SQL", "Amazon Dataset", "Data Aggregation", "Joins", "Subqueries", "Business Intelligence"],
        description: "Comprehensive e-commerce sales analysis using SQL on Amazon Sales Dataset to uncover key business insights, sales performance trends, and revenue optimization opportunities.",
        features: [
            "Raw data cleaning and structuring with SQL",
            "Total sales, profit, and quantity analysis",
            "Top 5 cities and states revenue identification",
            "Monthly and yearly trend analysis",
            "Aggregate functions for meaningful KPIs",
            "Product performance categorization",
            "Discount and shipping impact analysis",
            "Business intelligence dashboard queries"
        ],
        technicalDetails: {
            framework: "MySQL Workbench",
            dataset: "Amazon Sales Dataset",
            queryTypes: "Aggregate, Joins, Subqueries, Window Functions",
            dataVolume: "50K+ sales transactions",
            performance: "Optimized query execution",
            insights: "Month-over-month growth analysis",
            kpis: "Revenue, Profit Margin, Customer Metrics"
        },
        results: {
            salesGrowth: "Consistent month-over-month increase",
            topStates: "California and New York drive major revenue",
            categoryPerformance: "Electronics and Fashion highest demand",
            profitabilityFactors: "Discount and shipping time influence profitability"
        },
        applications: [
            "Business intelligence and reporting",
            "Sales performance optimization",
            "Inventory management insights",
            "Customer behavior analysis",
            "Revenue forecasting and planning"
        ],
        futurePlans: [
            "Integration with real-time data pipelines",
            "Advanced analytics with Python integration",
            "Predictive modeling for sales forecasting",
            "Interactive dashboard development"
        ]
    },
    // Analytics Project 12/20 - Financial Portfolio Analysis using Python
    {
        id: 112,
        title: "Financial Portfolio Analysis using Python",
        category: "analytics",
        image: "port/analytics/12.jpg",
        technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "yFinance", "Portfolio Optimization", "Financial Analysis"],
        description: "Comprehensive financial portfolio analysis using Python to evaluate performance of diversified stock portfolio (AMZN, GOOGL, MSFT, AAPL) with real trading data from January 2024 to January 2025.",
        features: [
            "Real-time stock data extraction using yFinance",
            "Portfolio performance analysis with weighted returns",
            "Daily closing price visualization and trend analysis",
            "Annual return calculation with multiple weighting strategies",
            "Risk assessment and volatility analysis",
            "Portfolio optimization for maximum returns",
            "Growth trend visualization with Matplotlib",
            "Diversification impact analysis"
        ],
        technicalDetails: {
            framework: "Python + Financial Libraries",
            dataSource: "yFinance API for real-time stock data",
            dataPeriod: "2024-01-01 to 2025-01-01",
            portfolioWeights: "[0.4, 0.3, 0.2, 0.1] for AMZN, GOOGL, MSFT, AAPL",
            analysis: "Daily closing prices and portfolio growth calculation",
            visualization: "Matplotlib for trend plotting and performance charts",
            optimization: "Portfolio weighting strategies comparison"
        },
        results: {
            optimalReturn: "35.4% annual return with optimal weighting",
            conservativeReturn: "20.24% with conservative investment mix",
            topPerformer: "Amazon (AMZN) showed highest growth impact",
            volatility: "Moderate portfolio volatility with balanced performance",
            diversificationBenefit: "Steady upward momentum across tech leaders"
        },
        applications: [
            "Investment portfolio management and optimization",
            "Risk assessment and financial planning",
            "Algorithmic trading strategy development",
            "Wealth management and advisory services",
            "Financial performance reporting and analytics"
        ],
        futurePlans: [
            "Integration with real-time trading APIs",
            "Machine learning for predictive portfolio optimization",
            "Risk management with VaR and Sharpe ratio analysis",
            "Interactive dashboard for portfolio monitoring"
        ]
    },
    // Analytics Project 13/20 - Health Care Analysis Dashboard using Power BI
    {
        id: 113,
        title: "Health Care Analysis Dashboard using Power BI",
        category: "analytics",
        image: "port/analytics/13.jpg",
        technologies: ["Power BI", "Power Query", "DAX", "Data Modeling", "Healthcare Analytics", "Business Intelligence"],
        description: "Interactive healthcare analysis dashboard using Power BI to visualize patient admissions, billing trends, and medical conditions across multiple years for data-driven healthcare management.",
        features: [
            "Patient admission tracking from 2019-2024",
            "Billing trend analysis with average cost calculations",
            "Gender distribution visualization and demographics",
            "Insurance provider analysis and coverage mapping",
            "Medical condition billing breakdown and insights",
            "Interactive dashboard with drill-down capabilities",
            "Year-over-year growth analysis and forecasting",
            "Cost optimization insights for healthcare administrators"
        ],
        technicalDetails: {
            framework: "Power BI Desktop + Power Query",
            dataModeling: "Star schema with fact and dimension tables",
            calculations: "DAX formulas for KPIs and measures",
            visualization: "Interactive charts, KPI cards, and filters",
            dataVolume: "55.50K patient records analyzed",
            timePeriod: "2019-2024 healthcare data analysis",
            performance: "Optimized for real-time dashboard updates"
        },
        results: {
            totalPatients: "55.50K patients analyzed",
            averageBilling: "₹25.54K per patient",
            peakAdmissions: "11K patients annually (2022-2023)",
            genderDistribution: "50.04% Female, 49.96% Male",
            topConditions: "Diabetes (₹238.54M), Obesity (₹238.21M), Arthritis (₹237.33M)",
            insuranceProviders: "Cigna, Medicare, UnitedHealthcare, Blue Cross, Aetna"
        },
        applications: [
            "Healthcare administration and management",
            "Patient flow optimization and resource planning",
            "Insurance claim analysis and cost control",
            "Medical condition trend monitoring",
            "Healthcare business intelligence and reporting"
        ],
        futurePlans: [
            "Integration with real-time hospital management systems",
            "Predictive analytics for patient admission forecasting",
            "Advanced cost optimization algorithms",
            "Mobile dashboard deployment for healthcare executives"
        ]
    },
    },
    {
        id: 115,
        title: "India Crime Analysis Dashboard using Power BI",
        category: "analytics",
        description: "Comprehensive Power BI dashboard analyzing crime statistics across India with focus on gender distribution, crime types, weapon usage, and police deployment trends. Features interactive visualizations showing victim demographics, case closure rates, crime trends from 2020-2024, and weapon-specific police deployment patterns.",
        technologies: ["Power BI", "DAX", "Data Cleaning", "Interactive Dashboards", "Drillthrough Filters"],
        frameworks: ["Power BI Desktop", "Power Query", "DAX Studio"],
        accuracy: "95% data accuracy",
        results: "Total 40,160 victims analyzed (22,423 female, 13,405 male, 4,332 others), 50/50 case closure rate, crime decline in 2024, highest police deployment for knife incidents (58,965)",
        applications: ["Crime Prevention", "Policy Planning", "Public Safety", "Law Enforcement Analytics", "Gender-based Crime Analysis"],
        futurePlans: "Integrate real-time crime data feeds, add predictive crime hotspot modeling, implement automated alert systems for high-risk areas",
        imageUrl: "port/analytics/15.jpg",
        videoUrl: "port/analytics/15.mp4",
        githubUrl: "https://github.com/yourusername/india-crime-analysis-powerbi",
        liveUrl: "https://app.powerbi.com/view?r=eyJrIjoiY3JpbWVhbmFseXNpcyJ9"
    }
];

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}
        },
        {
            id: 116,
            title: "Restaurant Food Sales Dashboard",
            category: "analytics",
            image: "port/analytics/16.jpg",
            technologies: ["Tableau", "Data Visualization", "KPI Cards", "Trend Analysis", "Business Intelligence"],
            frameworks: ["Tableau Desktop", "Tableau Public", "Data Cleaning"],
            description: "Interactive Tableau dashboard analyzing restaurant food sales performance across profit, sales trends, item types, and customer purchase behavior with comprehensive business insights.",
            features: [
                "Total Sales & Profit KPI Cards (₹2,75,230 sales, ₹2,25,689 profit)",
                "Item Type Performance Analysis (Fast Food vs Beverages)",
                "Top Performing Items Dashboard (Sandwiches, Cold Coffee, Frankie)",
                "Monthly Sales Trend Visualization with Peak Analysis",
                "Time-based Sales Analysis (Afternoon & Night Peak Periods)",
                "Yearly Trend Comparison (2022-2023 Performance)",
                "Interactive Filters and Drill-down Capabilities",
                "Profit Contribution Analysis by Category"
            ],
            results: {
                accuracy: "100% Data Accuracy",
                performance: "Real-time Dashboard Updates",
                insights: "Fast Food 65% Profit Share, Afternoon/Night 40% Sales Peak"
            },
            applications: [
                "Restaurant Operations Optimization",
                "Inventory Planning & Demand Forecasting",
                "Sales Strategy Development",
                "Customer Behavior Analysis",
                "Promotional Campaign Planning",
                "Business Performance Monitoring"
            ],
            futurePlans: [
                "Customer Segmentation Analysis",
                "Predictive Sales Forecasting",
                "Real-time Integration with POS Systems",
                "Mobile Dashboard Development"
            ]
        }
    ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}
        },
        {
            id: 117,
            title: "London House Price Analysis Dashboard",
            category: "analytics",
            image: "port/analytics/17.jpg",
            technologies: ["Power BI", "DAX", "Data Modeling", "Scatter Plots", "Trend Lines", "KPI Cards"],
            frameworks: ["Power BI Desktop", "Power Query", "DAX Studio"],
            description: "Interactive Power BI dashboard analyzing London house prices across localities, exploring how factors like carpet area, tax rate, and room count influence property sale prices with comprehensive market insights.",
            features: [
                "Locality-wise Sale Price Contribution Analysis (Greenwich, Bridgeport, Fairfield)",
                "Yearly Price Trend Visualization (2009-2022 Market Recovery)",
                "Price Reality Check - Affordability Indicator by Locality",
                "Rooms vs Sale Price Correlation Analysis",
                "Carpet Area Impact on Property Valuations (800-1500 sq.ft sweet spot)",
                "Property Tax Rate vs Valuation Relationship",
                "Interactive Filters and Cross-filtering Capabilities",
                "Premium Zone Identification through Tax Rate Analysis"
            ],
            results: {
                accuracy: "100% Data Accuracy",
                performance: "Real-time Dashboard Interactions",
                insights: "Greenwich 0.63 Affordability, Post-2016 Market Recovery, 800-1500 sq.ft Premium Range"
            },
            applications: [
                "Real Estate Investment Decision Support",
                "Property Buyer Market Analysis",
                "Housing Market Trend Forecasting",
                "Locality Affordability Assessment",
                "Investment Portfolio Optimization",
                "Real Estate Agent Market Intelligence"
            ],
            futurePlans: [
                "Predictive Price Modeling Integration",
                "Mortgage Calculator Integration",
                "Real-time Market Data Feeds",
                "Mobile Dashboard Development"
            ]
        }
    ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}
        },
        {
            id: 118,
            title: "Endangered Species Analysis using Python",
            category: "analytics",
            image: "port/analytics/18.jpg",
            technologies: ["Python", "Pandas", "NumPy", "Data Cleaning", "Grouping & Aggregation", "EDA"],
            frameworks: ["Pandas", "NumPy", "Matplotlib", "Seaborn"],
            description: "Comprehensive analysis of global wildlife conservation status using Python to understand endangered species distribution and identify critical conservation priorities across 80 species from the Endangered Species Report.",
            features: [
                "Conservation Status Distribution Analysis (97 → 80 species after cleaning)",
                "Critically Endangered Species Identification (16 species including Amur Leopard, Black Rhino)",
                "Endangered Species Breakdown (29 species including African Elephant, Blue Whale)",
                "Vulnerable & Near Threatened Classification (23 species total)",
                "Marine Species Risk Pattern Analysis (Whales, Dolphins, Turtles)",
                "Data Imbalance Assessment (Least Concern vs Threatened Groups)",
                "Species Scientific & Common Name Mapping",
                "Conservation Priority Ranking System"
            ],
            results: {
                accuracy: "100% Data Processing Accuracy",
                performance: "80/97 Species Successfully Analyzed",
                insights: "29 Endangered, 16 Critically Endangered, Marine Animals High Risk Pattern"
            },
            applications: [
                "Wildlife Conservation Policy Development",
                "Biodiversity Protection Planning",
                "Species Monitoring Program Design",
                "Conservation Funding Allocation",
                "Environmental Impact Assessment",
                "Global Conservation Status Reporting"
            ],
            futurePlans: [
                "Predictive Conservation Status Modeling",
                "Geographic Distribution Analysis",
                "Population Trend Forecasting",
                "Interactive Conservation Dashboard"
            ]
        }
    ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}
        },
        {
            id: 119,
            title: "IPL 2025 Analysis Dashboard",
            category: "analytics",
            image: "port/analytics/19.jpg",
            technologies: ["Power BI", "DAX", "Data Modeling", "Interactive Visuals", "Sports Analytics"],
            frameworks: ["Power BI Desktop", "Power Query", "DAX Studio"],
            description: "Comprehensive IPL 2025 season analysis dashboard using Power BI covering match outcomes, team performances, toss impact, and championship insights across 74 matches with RCB as season champions.",
            features: [
                "Season Champion Analysis (RCB Title Winners)",
                "Match Statistics Overview (74 total matches, 3 no results)",
                "Target Analysis (Highest: 286, Lowest: 95)",
                "Top Performers Tracking (759 runs Orange Cap, 25 wickets Purple Cap)",
                "Team Performance Breakdown (PBKS 11 wins, RCB 10 wins)",
                "Toss Impact Analysis (Toss Winner vs Match Winner correlation)",
                "Bowling First Preference Insights",
                "Interactive Team Comparison Visuals"
            ],
            results: {
                accuracy: "100% Match Data Coverage",
                performance: "Real-time Dashboard Updates",
                insights: "RCB Champions, PBKS Most Wins (11), Bowling First Advantage"
            },
            applications: [
                "Cricket Team Strategy Analysis",
                "Player Performance Evaluation",
                "Match Outcome Prediction Insights",
                "Toss Decision Strategy Optimization",
                "Fantasy Cricket Analytics",
                "Sports Broadcasting Intelligence"
            ],
            futurePlans: [
                "Player Performance Prediction Models",
                "Real-time Match Analytics Integration",
                "Advanced Team Strategy Recommendations",
                "Mobile Sports Analytics App"
            ]
        }
    ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}
        },
        {
            id: 120,
            title: "Job Market Analysis Using Python",
            category: "analytics",
            image: "port/analytics/20.jpg",
            technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Statistical Analysis"],
            frameworks: ["Pandas", "NumPy", "Matplotlib", "Seaborn"],
            description: "Comprehensive AI-driven job market analysis using Python exploring job titles, industry trends, salaries, skill requirements, automation risk, AI adoption levels, and remote work patterns across multiple industries.",
            features: [
                "Complete Data Analysis Cycle (Cleaning, EDA, Visualization)",
                "Top Job Roles Analysis (Data Scientist, Software Engineer, AI Researcher)",
                "Industry Hiring Trends (Technology, Finance, Healthcare dominance)",
                "Skills Demand Analysis (Python, ML, Data Analysis, Cybersecurity)",
                "Salary Distribution Study (Average $91K, AI Research highest)",
                "AI Adoption Level Assessment (Medium 36%, Low 35%, High 29%)",
                "Automation Risk Evaluation by Industry",
                "Remote Work Trends Analysis (50% remote-friendly roles)"
            ],
            results: {
                accuracy: "100% Data Processing Accuracy",
                performance: "Complete Job Market Intelligence",
                insights: "Tech/Finance Lead, Python/ML Core Skills, 50% Remote-Friendly"
            },
            applications: [
                "Career Planning and Development",
                "Recruitment Strategy Optimization",
                "Skills Gap Analysis for Organizations",
                "Salary Benchmarking Studies",
                "AI Transformation Impact Assessment",
                "Remote Work Policy Development"
            ],
            futurePlans: [
                "Predictive Job Market Modeling",
                "Real-time Job Trend Monitoring",
                "Personalized Career Recommendation System",
                "Interactive Job Market Dashboard"
            ]
        }
    ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = projectDatabase;
}