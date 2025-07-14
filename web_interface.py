from flask import Flask, render_template, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import json
import os
import threading
from datetime import datetime
from advanced_nhl_predictor import NHLGamePredictor

app = Flask(__name__)

# Initialize predictor
predictor = None

# Enhanced NHL Team Data with detailed information
NHL_TEAMS = {
    1: {"name": "New Jersey Devils", "city": "New Jersey", "conference": "Eastern", "division": "Metropolitan", "color": "#CE1126"},
    2: {"name": "New York Islanders", "city": "New York", "conference": "Eastern", "division": "Metropolitan", "color": "#00539B"},
    3: {"name": "New York Rangers", "city": "New York", "conference": "Eastern", "division": "Metropolitan", "color": "#0038A8"},
    4: {"name": "Philadelphia Flyers", "city": "Philadelphia", "conference": "Eastern", "division": "Metropolitan", "color": "#F74902"},
    5: {"name": "Pittsburgh Penguins", "city": "Pittsburgh", "conference": "Eastern", "division": "Metropolitan", "color": "#FCB514"},
    6: {"name": "Boston Bruins", "city": "Boston", "conference": "Eastern", "division": "Atlantic", "color": "#FFB81C"},
    7: {"name": "Buffalo Sabres", "city": "Buffalo", "conference": "Eastern", "division": "Atlantic", "color": "#003087"},
    8: {"name": "Montreal Canadiens", "city": "Montreal", "conference": "Eastern", "division": "Atlantic", "color": "#AF1E2D"},
    9: {"name": "Ottawa Senators", "city": "Ottawa", "conference": "Eastern", "division": "Atlantic", "color": "#C52032"},
    10: {"name": "Toronto Maple Leafs", "city": "Toronto", "conference": "Eastern", "division": "Atlantic", "color": "#003E7E"},
    12: {"name": "Carolina Hurricanes", "city": "Carolina", "conference": "Eastern", "division": "Metropolitan", "color": "#CE1126"},
    13: {"name": "Florida Panthers", "city": "Florida", "conference": "Eastern", "division": "Atlantic", "color": "#041E42"},
    14: {"name": "Tampa Bay Lightning", "city": "Tampa Bay", "conference": "Eastern", "division": "Atlantic", "color": "#002868"},
    15: {"name": "Washington Capitals", "city": "Washington", "conference": "Eastern", "division": "Metropolitan", "color": "#041E42"},
    16: {"name": "Chicago Blackhawks", "city": "Chicago", "conference": "Western", "division": "Central", "color": "#CF0A2C"},
    17: {"name": "Detroit Red Wings", "city": "Detroit", "conference": "Eastern", "division": "Atlantic", "color": "#CE1126"},
    18: {"name": "Nashville Predators", "city": "Nashville", "conference": "Western", "division": "Central", "color": "#FFB81C"},
    19: {"name": "St. Louis Blues", "city": "St. Louis", "conference": "Western", "division": "Central", "color": "#002F87"},
    20: {"name": "Calgary Flames", "city": "Calgary", "conference": "Western", "division": "Pacific", "color": "#C8102E"},
    21: {"name": "Colorado Avalanche", "city": "Colorado", "conference": "Western", "division": "Central", "color": "#6F263D"},
    22: {"name": "Edmonton Oilers", "city": "Edmonton", "conference": "Western", "division": "Pacific", "color": "#041E42"},
    23: {"name": "Vancouver Canucks", "city": "Vancouver", "conference": "Western", "division": "Pacific", "color": "#001F5B"},
    24: {"name": "Anaheim Ducks", "city": "Anaheim", "conference": "Western", "division": "Pacific", "color": "#F47A38"},
    25: {"name": "Dallas Stars", "city": "Dallas", "conference": "Western", "division": "Central", "color": "#006847"},
    26: {"name": "Los Angeles Kings", "city": "Los Angeles", "conference": "Western", "division": "Pacific", "color": "#111111"},
    28: {"name": "San Jose Sharks", "city": "San Jose", "conference": "Western", "division": "Pacific", "color": "#006D75"},
    29: {"name": "Columbus Blue Jackets", "city": "Columbus", "conference": "Eastern", "division": "Metropolitan", "color": "#002654"},
    30: {"name": "Minnesota Wild", "city": "Minnesota", "conference": "Western", "division": "Central", "color": "#154734"},
    52: {"name": "Winnipeg Jets", "city": "Winnipeg", "conference": "Western", "division": "Central", "color": "#041E42"},
    53: {"name": "Arizona Coyotes", "city": "Arizona", "conference": "Western", "division": "Central", "color": "#8C2633"},
    54: {"name": "Vegas Golden Knights", "city": "Vegas", "conference": "Western", "division": "Pacific", "color": "#B4975A"},
    55: {"name": "Seattle Kraken", "city": "Seattle", "conference": "Western", "division": "Pacific", "color": "#99D9D9"}
}

# Legacy teams mapping for compatibility
TEAMS = {id: data["name"] for id, data in NHL_TEAMS.items()}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏒 Advanced NHL Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .innovation-badge {
            background: linear-gradient(45deg, #ff6b6b, #ffa726);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            display: inline-block;
            margin-top: 10px;
        }
        
        .prediction-panel {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .team-selector {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 30px;
            align-items: center;
            margin-bottom: 40px;
        }
        
        .team-column {
            text-align: center;
        }
        
        .team-select {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: white;
            transition: all 0.3s ease;
        }
        
        .team-select:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .vs-badge {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 50%;
            font-size: 24px;
            font-weight: bold;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .home-away-toggle {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .toggle-btn {
            padding: 12px 24px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .toggle-btn.active {
            background: #667eea;
            color: white;
        }
        
        .predict-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            display: none;
            animation: fadeIn 0.6s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .result-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .result-card.winner {
            border-color: #28a745;
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
        }
        
        .team-name {
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        .win-prob {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .goals-pred {
            font-size: 1.6em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .confidence-section {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .confidence-bar {
            background: #e0e0e0;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ffc107, #28a745);
            transition: width 0.8s ease;
        }
        
        .advanced-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .metric {
            text-align: center;
            padding: 15px;
            background: #f1f3f4;
            border-radius: 10px;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .upcoming-games {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .upcoming-games h3 {
            text-align: center;
            margin-bottom: 25px;
            color: #333;
            font-size: 1.5em;
        }
        
        .game-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .game-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .team-selector {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            
            .result-grid {
                grid-template-columns: 1fr;
            }
            
            .advanced-metrics {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏒 Advanced NHL Predictor</h1>
            <p>Revolutionary AI-powered hockey predictions with advanced analytics</p>
            <div class="innovation-badge">🚀 Transformer Networks & Ensemble ML</div>
        </div>
        
        .content {
            padding: 40px;
        }
        
        .prediction-form {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .form-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            align-items: center;
        }
        
        .form-group {
            flex: 1;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        select, button {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        select:focus {
            border-color: #2a5298;
            outline: none;
            box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.1);
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .vs-indicator {
            font-size: 24px;
            font-weight: bold;
            color: #2a5298;
            text-align: center;
            margin: 0 10px;
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
            display: none;
        }
        
        .result-header {
            text-align: center;
            margin-bottom: 25px;
        }
        
        .result-header h3 {
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .matchup {
            font-size: 1.3em;
            color: #34495e;
        }
        
        .prediction-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .prediction-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .prediction-card h4 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .prediction-value {
            font-size: 2em;
            font-weight: bold;
            color: #2a5298;
            margin-bottom: 10px;
        }
        
        .prediction-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .recommendation {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .upcoming-games {
            margin-top: 40px;
        }
        
        .upcoming-games h3 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        
        .game-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .game-teams {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .game-prediction {
            color: #2a5298;
            font-weight: 600;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #7f8c8d;
        }
        
        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        @media (max-width: 768px) {
            .form-row {
                flex-direction: column;
            }
            
            .vs-indicator {
                transform: rotate(90deg);
                margin: 15px 0;
            }
            
            .prediction-cards {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏒 Advanced NHL Predictor</h1>
            <p>State-of-the-art machine learning for hockey predictions</p>
        </div>
        
        <div class="prediction-panel">
            <div class="team-selector">
                <div class="team-column">
                    <h3 style="margin-bottom: 15px; color: #333;">Away Team</h3>
                    <select id="team1" class="team-select">
                        <option value="">Select away team...</option>
                        {% for team_key, team_data in teams_data.items() %}
                        <option value="{{ team_data.id }}">{{ team_data.name }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="vs-badge">VS</div>
                
                <div class="team-column">
                    <h3 style="margin-bottom: 15px; color: #333;">Home Team</h3>
                    <select id="team2" class="team-select">
                        <option value="">Select home team...</option>
                        {% for team_key, team_data in teams_data.items() %}
                        <option value="{{ team_data.id }}">{{ team_data.name }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>
            
            <div class="home-away-toggle">
                <button class="toggle-btn active" onclick="setVenue('neutral')">🏟️ Neutral</button>
                <button class="toggle-btn" onclick="setVenue('home')">🏠 Home Advantage</button>
                <button class="toggle-btn" onclick="setVenue('playoff')">🏆 Playoff Mode</button>
            </div>
            
            <button onclick="predictGame()" id="predictBtn" class="predict-btn">
                🤖 Predict with AI
            </button>
            
            <div id="predictionResult" class="results">
                <!-- Results will be populated here -->
            </div>
        </div>
        
        <div class="upcoming-games">
            <h3>� Upcoming Games Predictions</h3>
            <button onclick="loadUpcomingGames()" id="upcomingBtn" class="predict-btn" style="margin-bottom: 20px;">
                � Load Next Games
            </button>
            <div id="upcomingGames">
                <!-- Upcoming games will be populated here -->
            </div>
        </div>
    </div>

    <script>
        let currentVenue = 'neutral';
        
        function setVenue(venue) {
            currentVenue = venue;
            document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        async function predictGame() {
            const team1 = document.getElementById('team1').value;
            const team2 = document.getElementById('team2').value;
            const btn = document.getElementById('predictBtn');
            const result = document.getElementById('predictionResult');
            
            if (!team1 || !team2) {
                alert('Please select both teams');
                return;
            }
            
            if (team1 === team2) {
                alert('Please select different teams');
                return;
            }
            
            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Analyzing with AI...';
            result.style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        team1_id: parseInt(team1),
                        team2_id: parseInt(team2),
                        venue: currentVenue
                    })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayPrediction(data);
                result.style.display = 'block';
                
            } catch (error) {
                result.innerHTML = `<div style="color: #e74c3c; text-align: center; padding: 20px;">❌ Error: ${error.message}</div>`;
                result.style.display = 'block';
            }
            
            btn.disabled = false;
            btn.innerHTML = '🤖 Predict with AI';
        }
        
        function displayPrediction(data) {
            const result = document.getElementById('predictionResult');
            
            const team1WinProb = (data.predictions.team1_win_probability * 100).toFixed(1);
            const team2WinProb = (100 - team1WinProb).toFixed(1);
            const team1Goals = data.predictions.team1_predicted_goals?.toFixed(1) || '2.5';
            const team2Goals = data.predictions.team2_predicted_goals?.toFixed(1) || '2.3';
            const margin = data.predictions.predicted_goal_margin?.toFixed(1) || '0.2';
            const confidence = (data.predictions.confidence?.win_prediction * 100)?.toFixed(0) || '78';
            
            const isTeam1Winner = parseFloat(team1WinProb) > 50;
            
            result.innerHTML = `
                <div class="result-grid">
                    <div class="result-card ${isTeam1Winner ? 'winner' : ''}">
                        <div class="team-name">${data.team1.name}</div>
                        <div class="win-prob">${team1WinProb}%</div>
                        <div class="goals-pred">⚽ ${team1Goals} goals</div>
                        <div style="font-size: 0.9em; color: #666;">Away Team</div>
                    </div>
                    
                    <div class="result-card ${!isTeam1Winner ? 'winner' : ''}">
                        <div class="team-name">${data.team2.name}</div>
                        <div class="win-prob">${team2WinProb}%</div>
                        <div class="goals-pred">⚽ ${team2Goals} goals</div>
                        <div style="font-size: 0.9em; color: #666;">Home Team</div>
                    </div>
                </div>
                
                <div class="confidence-section">
                    <h4 style="margin-bottom: 15px; text-align: center;">🎯 Prediction Confidence</h4>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                    </div>
                    <p style="text-align: center; margin: 0; font-weight: bold;">${confidence}% Confidence</p>
                </div>
                
                <div class="advanced-metrics">
                    <div class="metric">
                        <div class="metric-label">Goal Margin</div>
                        <div class="metric-value">${margin > 0 ? '+' : ''}${margin}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Goals</div>
                        <div class="metric-value">${(parseFloat(team1Goals) + parseFloat(team2Goals)).toFixed(1)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Model Type</div>
                        <div class="metric-value">Transformer</div>
                    </div>
                </div>
                
                <div style="background: #e8f5e8; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
                    <strong>💡 ${data.recommendation || 'Advanced AI analysis complete!'}</strong>
                </div>
            `;
        }
        
        async function loadUpcomingGames() {
            const btn = document.getElementById('upcomingBtn');
            const container = document.getElementById('upcomingGames');
            
            btn.disabled = true;
            btn.innerHTML = '<div class="spinner"></div> Loading...';
            container.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading upcoming games...</p></div>';
            
            try {
                const response = await fetch('/upcoming');
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="loading">📅 No upcoming games found</div>';
                    return;
                }
                
                let html = '<div style="display: grid; gap: 15px;">';
                data.slice(0, 8).forEach(game => {
                    const winProb = (game.predictions.team1_win_probability * 100).toFixed(0);
                    const team1Name = game.team1.name;
                    const team2Name = game.team2.name;
                    const isStrong = winProb > 65 || winProb < 35;
                    
                    html += `
                        <div class="game-item" onclick="quickPredict('${game.team1.id}', '${game.team2.id}')">
                            <div>
                                <strong>${team1Name} @ ${team2Name}</strong>
                                <div style="font-size: 0.9em; color: #666;">${game.date || 'Upcoming'}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: bold; color: ${isStrong ? '#28a745' : '#6c757d'};">
                                    ${winProb}% ${team1Name}
                                </div>
                                <div style="font-size: 0.8em; color: #666;">
                                    ${isStrong ? '🔥 Strong' : '⚖️ Close'}
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                
                container.innerHTML = html;
                
            } catch (error) {
                container.innerHTML = `<div style="color: #e74c3c; text-align: center; padding: 20px;">❌ Error: ${error.message}</div>`;
            }
            
            btn.disabled = false;
            btn.innerHTML = '📅 Load Next Games';
        }
        
        function quickPredict(team1Id, team2Id) {
            document.getElementById('team1').value = team1Id;
            document.getElementById('team2').value = team2Id;
            predictGame();
        }
        
        // Auto-load upcoming games on page load
        window.addEventListener('load', () => {
            setTimeout(loadUpcomingGames, 1500);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, teams_data=NHL_TEAMS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        team1_id = data['team1_id']
        team2_id = data['team2_id']
        venue = data.get('venue', 'neutral')
        
        if not predictor or not predictor.is_trained:
            return jsonify({'error': 'Model not trained yet'})
        
        # Determine home advantage based on venue
        is_team1_home = venue == 'home'
        
        prediction = predictor.predict_game(
            team1_id=team1_id,
            team2_id=team2_id,
            is_team1_home=is_team1_home
        )
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upcoming')
def upcoming():
    try:
        if not predictor or not predictor.is_trained:
            return jsonify({'error': 'Model not trained yet'})
        
        predictions = predictor.predict_upcoming_games(days_ahead=5)
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/status')
def status():
    return jsonify({
        'model_trained': predictor.is_trained if predictor else False,
        'features_count': len(predictor.feature_columns) if predictor and predictor.feature_columns else 0,
        'teams_available': len(NHL_TEAMS),
        'model_type': 'Advanced Ensemble with Transformers',
        'accuracy': predictor.model_metrics.get('accuracy', 0) if predictor and hasattr(predictor, 'model_metrics') else 0
    })

def initialize_predictor():
    """Initialize and train the predictor"""
    global predictor
    
    print("🏒 Initializing Advanced NHL Predictor...")
    
    try:
        predictor = NHLGamePredictor("data")
        
        # Check if we have a saved model
        model_path = "models/nhl_predictor_advanced.pkl"
        if os.path.exists(model_path):
            print("📁 Loading saved model...")
            predictor.load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print("🏗️ Training new model...")
            df = predictor.load_and_preprocess_data(use_api=True, api_days=90)
            metrics = predictor.train_model(df, test_size=0.2)
            
            # Save the model
            os.makedirs("models", exist_ok=True)
            predictor.save_model(model_path)
            print("✅ Model trained and saved!")
            
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        predictor = None

if __name__ == '__main__':
    initialize_predictor()
    
    if predictor and predictor.is_trained:
        print("\n🚀 Starting web interface...")
        print("🌐 Access the predictor at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Could not start web interface - model not available")
