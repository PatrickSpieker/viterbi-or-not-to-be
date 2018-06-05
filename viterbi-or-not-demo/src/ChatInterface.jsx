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
            return !message.hasOwnProperty('action') && message.message.trim() !== '';
        });

        let range = {}
        let min = {}
        for (let feature of Object.keys(this.props.features)) {
            min[feature] = Math.min.apply(Math, this.props.features[feature]);
            range[feature] = Math.max.apply(Math, this.props.features[feature]) - min[feature];
        }

        if (chatMessages.length > 0) {
            let message = chatMessages[0];
            let lastAuthor = message.author;

            let self = (message.author === this.props.author) ? ('self') : ('other')
            messageLog.push(
                <li className={self + ' label'} key={message.key + '_new_author'}>{message.author}</li>
            )

            for (let i = 0; i < chatMessages.length; i++) {
                let message = chatMessages[i];
                let scaledPrediction = 0.6 * this.props.predictions[i] + 0.2;
                let background = this.props.predictions.length === 0 ? {} : {backgroundColor: 'rgba(5, 168, 170, ' + scaledPrediction + ')'};

                if (message.author !== lastAuthor) {
                    // New author, needs to have name printed
                    self = (message.author === this.props.author) ? ('self') : ('other')
                    messageLog.push(
                        <li className={self + ' label'} key={message.key + '_new_author'}>{message.author}</li>
                    )
                    lastAuthor = message.author;
                }

                self = (message.author === this.props.author) ? ('self') : ('other')
                let statistics = [];

                if (this.props.predictions.length !== 0) {
                    for (let feature of this.props.selectedFeatures) {
                        console.log('dog ! pup !');
                        console.log(feature);
                        console.log(this.props.selectedFeatures);
                        let featureValue = this.props.features[feature][i];
                        let height = ((featureValue - min[feature]) / range[feature] * 1.75);
                        let bar = (<div className="feature-graph" style={{height: height + 'rem'}}></div>);
                        statistics.push(bar);
                    }
                }

                messageLog.push(
                    <li className={self + ' message'} key={message.key}>
                        <p className="message-content" style={background}>{message.message}</p>
                        <div className="message-features">{statistics}</div>
                    </li>
                )
            }
        }

        let interfaceClass = this.props.predictions.length === 0 ? '' : 'highlighted';

        return (
            <div id="chat-interface" className={interfaceClass}>
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