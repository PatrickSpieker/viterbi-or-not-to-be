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
            summary: [],
            loading: false
        }

        this.sendMessage = this.sendMessage.bind(this);
        this.refreshSummary = this.refreshSummary.bind(this);
        this.closeSummary = this.closeSummary.bind(this);
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
        let message = {
            author: this.props.username,
            timestamp: Date.now(),
            message: messageText
        };
        this.props.db.collection(this.props.room).add(message);
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
            messageText.push(message.message);
            authors.push(message.author);
        });

        // Send fetch request
        fetch('http://viterb.me/api', {
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
                    summaryLines.push(messageText[i]);
                }
            }

            this.setState({
                loading: false,
                summary: summaryLines,
                predictions: predictions
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
                    <ChatInterface chatMessages={this.state.chatMessages} predictions={this.state.predictions} sendMessage={this.sendMessage} />
                </div>
                <div id="summary-container" className={this.state.summary.length === 0 ? 'closed' : 'open'}>
                    <SummaryInterface summary={this.state.summary} predictions={this.state.predictions} />
                </div>
            </div>
        );
    }
}