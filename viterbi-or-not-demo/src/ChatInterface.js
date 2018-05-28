import React, {Component} from 'react';

export default class ChatInterface extends Component {
    constructor(props) {
        super(props);

        this.state = {
            message: ''
        }

        this.handleMessage = this.handleMessage.bind(this);
        this.handleSend = this.handleSend.bind(this);
    }

    componentDidUpdate() {
        var objDiv = document.getElementById('chat-listing');
        objDiv.scrollTop = objDiv.scrollHeight;
    }

    handleMessage(event) {
        this.setState({message: event.target.value});
    }

    handleSend(event) {
        event.preventDefault();
        this.props.sendMessage(this.state.message);
        this.setState({message: ''});
    }

    render() {
        let messageLog = [];

        let chatMessages = this.props.chatMessages.filter((message) => {
            return !message.hasOwnProperty('action');
        });

        if (chatMessages.length > 0) {
            let message = chatMessages[0];
            let lastAuthor = message.author;

            messageLog.push(
                (message.author === this.props.author) ?
                (<li className="self label" key={message.key + '_new_author'}>{message.author}</li>) :
                (<li className="other label" key={message.key + '_new_author'}>{message.author}</li>)
            )

            for (let i = 0; i < chatMessages.length; i++) {
                let message = chatMessages[i];

                if (message.author !== lastAuthor) {
                    // New author, needs to have name printed
                    messageLog.push(
                        (message.author === this.props.author) ?
                        (<li className="self label" key={message.key + '_new_author'}>{message.author}</li>) :
                        (<li className="other label" key={message.key + '_new_author'}>{message.author}</li>)
                    )
                    lastAuthor = message.author;
                }

                messageLog.push(
                    (message.author === this.props.author) ?
                    (<li className="self message" key={message.key}>{message.message}</li>) :
                    (<li className="other message" key={message.key}>{message.message}</li>)
                )
            }
        }

        return (
            <div id="chat-interface">
                <ul id="chat-listing">
                    {messageLog}
                </ul>
                <form onSubmit={this.handleSend}>
                    <div id="chat-form" className="form-line">
                        <input id="chat-input" className="form-input" type="text" autoComplete="off" value={this.state.message} onChange={this.handleMessage} />
                        <button className="submit-button" type="submit"><i className="material-icons">arrow_forward</i></button>
                    </div>
                </form>
            </div>
        );
    }
}