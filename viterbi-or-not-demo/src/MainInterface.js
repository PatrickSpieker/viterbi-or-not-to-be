import React, {Component} from 'react';

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
            this.setState({chatMessages: chatMessages});
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

    render() {
        return (
            <div>
                <h1>{this.props.room}</h1>
                <ChatInterface chatMessages={this.state.chatMessages} sendMessage={this.sendMessage} />
                <SummaryInterface summary={this.state.summary} />
            </div>
        );
    }
}