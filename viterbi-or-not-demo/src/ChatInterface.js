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

    handleMessage(event) {
        this.setState({message: event.target.value});
    }

    handleSend(event) {
        event.preventDefault();
        this.props.sendMessage(this.state.message);
        this.setState({message: ''});
    }

    render() {
        let messageLog = this.props.chatMessages.map((message) =>
            (<li key={message.key}><b>{message.author}</b>{message.message}</li>)
        );

        return (
            <div>
                <ul>
                    {messageLog}
                </ul>
                <form onSubmit={this.handleSend}>
                    <label>
                        Message:
                        <input type="text" value={this.state.message} onChange={this.handleMessage} />
                    </label>
                    <input type="submit" value="Send" />
                </form>
            </div>
        );
    }
}