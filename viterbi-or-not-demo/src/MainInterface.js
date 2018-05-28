import React, { Component } from 'react';

import ChatInterface from './ChatInterface';
import SummaryInterface from './SummaryInterface';

export default class MainInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
            chatMessages: [],
            summary: []
        }

        this.sendMessage = this.sendMessage.bind(this);
        this.refreshSummary = this.refreshSummary.bind(this);
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

    refreshSummary() {
        // Package the data for sending
        let messageText = []
        let authors = []

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
            console.log(responseJson);

            let summaryLines = []
            var indices = new Array(responseJson.length);
            for (var i = 0; i < responseJson.length; ++i) {
                indices[i] = i;
            }

            indices.sort(function (a, b) { return responseJson[a] < responseJson[b] ? 1 : responseJson[a] > responseJson[b] ? -1 : 0; });

            console.log(indices);

            let included = indices.slice(6)

            console.log(included);

            for (let i = 0; i < messageText.length; i++) {
                if (responseJson[i] >= 0.1) {
                    summaryLines.push(messageText[i]);
                }
            }
            // for (let i = 0; i < messageText.length; i++) {
            //     if (i in included) {
            //         summaryLines.push(messageText[i]);
            //     }
            // }
            this.setState({summary: summaryLines})
        });

    }

    render() {
        return (
            <div id="main-interface-container">
                <div id="chat-container">
                    <div id="title-bar">
                        <button id="back-button" onClick={this.props.clearRoom} >
                            <i className="material-icons">arrow_back</i>
                        </button>
                        <h1>{this.props.room}</h1>
                        <button id="summary-button" className="active-button" onClick={this.refreshSummary} >
                            <i className="material-icons">format_list_bulleted</i>
                            Summarize
                        </button>
                    </div>
                    <ChatInterface chatMessages={this.state.chatMessages} sendMessage={this.sendMessage} />
                </div>
                <div id="summary-container">
                    <SummaryInterface summary={this.state.summary} />
                </div>
            </div>
        );
    }
}