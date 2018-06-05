import React, { Component } from 'react';

import ChatInterface from './ChatInterface';
import SummaryInterface from './SummaryInterface';

import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.min.css';

const COMPRESS = 0.45;

export default class MainInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
            chatMessages: [],
            predictions: [],
            features: {},
            selectedFeatures: ['centroid_coherence', 'author_frequency'],
            selectedOrder: ['centroid_coherence', 'author_frequency'],
            summary: [],
            loading: false
        }

        this.sendMessage = this.sendMessage.bind(this);
        this.refreshSummary = this.refreshSummary.bind(this);
        this.closeSummary = this.closeSummary.bind(this);
        this.toggleFeature = this.toggleFeature.bind(this);
    }

    componentDidMount() {
        this.props.db.collection(this.props.room).onSnapshot((querySnapshot) => {
            let chatMessages = [];
            querySnapshot.forEach((doc) => {
                let message = doc.data();
                message.key = doc.id;
                chatMessages.push(message);
            });
            chatMessages.sort((a, b) => {
                return a.timestamp - b.timestamp
            });
            this.setState({ chatMessages: chatMessages });
        });
    }

    sendMessage(messageText) {
        if (messageText.trim() !== '') {
            let message = {
                author: this.props.username,
                timestamp: Date.now(),
                message: messageText
            };
            this.props.db.collection(this.props.room).add(message);
        }
    }

    toggleFeature(feature) {
        let newFeatures = this.state.selectedFeatures.slice();
        let newOrder = this.state.selectedOrder.slice();

        // Check if the feature to be toggled currently exists. If so,
        // delete it from both lists.
        if (this.state.selectedFeatures.includes(feature)) {
            let featureIndex = this.state.selectedFeatures.indexOf(feature);
            newFeatures.splice(featureIndex, 1);

            let orderIndex = this.state.selectedOrder.indexOf(feature);
            newOrder.splice(orderIndex, 1);
        }

        // If the feature isn't already selected, and there are fewer
        // than three features, simply add it.
        else if (this.state.selectedFeatures.length < 3) {
            newFeatures.push(feature);
            newOrder.push(feature);
        }

        // If the feature isn't already selected and there are three
        // features, replace the least recently selected one.
        else {
            let leastRecent = this.state.selectedOrder[0];
            let featureIndex = this.state.selectedFeatures.indexOf(leastRecent);

            newFeatures[featureIndex] = feature;
            newOrder = newOrder.slice(1);
            newOrder.push(feature);
        }

        if (newFeatures.length > 2 && newFeatures[2] === null) {
            console.log('sausage directory');
        }

        this.setState({
            selectedFeatures: newFeatures,
            selectedOrder: newOrder
        });
    }

    closeSummary() {
        this.setState({
            summary: [],
            predictions: []
        });
    }

    refreshSummary() {
        // Package the data for sending
        let messageText = []
        let authors = []

        this.setState({
            loading: true
        });

        this.state.chatMessages.forEach((message) => {
            if (message.message.trim() !== '') {
                messageText.push(message.message);
                authors.push(message.author);
            }
        });

        // Send fetch request
        // fetch('http://viterb.me/api', {
        fetch('http://127.0.0.1:5000/api', {
            method: 'POST',
            headers: {
                Accept: 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: messageText,
                authors: authors,
            })
        }).then((response) => response.json())
        .then((responseJson) => {
            // responseJson is an object containing various properties:
            //   predictions: an array in which each index corresponds to a message and each value corresponds to a score

            console.log(responseJson);

            let formattedLines = responseJson.formatted;
            let features = responseJson.features;
            let predictions = responseJson.predictions;
            let sortedPredictions = responseJson.predictions.slice(0);

            sortedPredictions.sort();

            // For the summary, retain the top COMPRESS % of sentences
            let preservedIndex = Math.round(sortedPredictions.length * (1 - COMPRESS));
            let thresholdPrediction = sortedPredictions[preservedIndex];

            console.log('Choosing threshold of ' + thresholdPrediction + ' to get a compression ratio of ' + COMPRESS);

            let summaryLines = [];
            for (let i = 0; i < predictions.length; i++) {
                if (predictions[i] >= thresholdPrediction) {
                    summaryLines.push(formattedLines[i]);
                }
            }

            this.setState({
                loading: false,
                summary: summaryLines,
                predictions: predictions,
                features: features
            });
        }).catch((error) => {
            toast.error('Could not connect to summarization API!');
            console.log(error);
            this.setState({
                loading: false
            });
        });
    }

    render() {
        let actionButton;
        if (this.state.loading) {
            actionButton = (
                <button id="summary-button" className="active-button" onClick={this.refreshSummary} >
                    <i className="material-icons infinite-spin">sync</i>
                </button>
            );
        } else if (this.state.summary.length === 0) {
            actionButton = (
                <button id="summary-button" className="active-button" onClick={this.refreshSummary} >
                    <i className="material-icons">format_list_bulleted</i>
                    Summarize
                </button>
            );
        } else {
            actionButton = (
                <button id="summary-button" className="active-button" onClick={this.closeSummary} >
                    <i className="material-icons">close</i>
                    Close Summary
                </button>
            );
        }

        return (
            <div id="main-interface-container">
                <div id="chat-container">
                    <div id="title-bar">
                        <button id="back-button" onClick={this.props.clearRoom} >
                            <i className="material-icons">arrow_back</i>
                        </button>
                        <h1>{this.props.room}</h1>
                        {actionButton}
                    </div>
                    <ChatInterface 
                        chatMessages={this.state.chatMessages}
                        predictions={this.state.predictions}
                        features={this.state.features}
                        selectedFeatures={this.state.selectedFeatures}
                        sendMessage={this.sendMessage} />
                </div>
                <div id="summary-container" className={this.state.summary.length === 0 ? 'closed' : 'open'}>
                    <SummaryInterface
                        summary={this.state.summary}
                        predictions={this.state.predictions} 
                        features={this.state.features}
                        selectedFeatures={this.state.selectedFeatures}
                        toggleFeature={this.toggleFeature} />
                </div>
            </div>
        );
    }
}