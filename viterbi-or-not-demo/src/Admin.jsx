import React, {Component} from 'react';

export default class Admin extends Component {
    constructor(props) {
        super(props);

        this.state = {
            conversation: '',
            room: ''
        }

        this.purgeChatrooms = this.purgeChatrooms.bind(this);
        this.uploadRoom = this.uploadRoom.bind(this);
        this.handleRoom = this.handleRoom.bind(this);
        this.handleConvo = this.handleConvo.bind(this);
    }

    purgeChatrooms() {
        this.props.db.collection('metadata').doc('reserved_rooms').get()
            .then(document => {
                if (document.exists) {
                    let takenCodes = document.data().reserved_rooms;
                    for (let code of takenCodes) {
                        this.props.db.collection(code.toString()).get()
                            .then(roomCollection => {
                                roomCollection.forEach(chatDoc => {
                                    console.log('deleting ' + code.toString() + ':' + chatDoc.id);
                                    this.props.db.collection(code.toString()).doc(chatDoc.id).delete();
                                });
                            });
                    }
                }
            });

        this.props.db.collection('metadata').doc('reserved_rooms').update({
            'reserved_rooms': []
        });
    }

    uploadRoom(event) {
        event.preventDefault();

        let timestamp = Date.now();

        for (let message of this.state.conversation.split('\n')) {
            let parts = message.split(':');
            console.log('adding ' + parts);
            this.props.db.collection(this.state.room).add({
                author: parts[0],
                timestamp: timestamp,
                message: parts[1].trim()
            });
            timestamp += 1;
        }
    }

    handleRoom(event) {
        this.setState({room: event.target.value});
    }

    handleConvo(event) {
        this.setState({conversation: event.target.value});
    }

    render() {
        return (
            <div>
                <button onClick={this.purgeChatrooms} type="button">Purge All Chatrooms</button>
                <form onSubmit={this.uploadRoom}>
                    <p>Room</p>
                    <input type="text" value={this.state.room} onChange={this.handleRoom} />
                    <p>Conversation (format is owner: hey walker)</p>
                    <textarea onChange={this.handleConvo}>{this.state.conversation}</textarea>
                    <input type="submit" />
                </form>
            </div>
        );
    }
}