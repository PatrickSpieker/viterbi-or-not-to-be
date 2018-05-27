import React, {Component} from 'react';

import RoomSelector from './RoomSelector';
import MainInterface from './MainInterface';

import firebase from 'firebase/app';
import 'firebase/firestore';

import { getConfig } from './environment.js';

export default class App extends Component {
    constructor(props) {
        super(props);

        let config = getConfig();
        firebase.initializeApp(config);

        const db = firebase.firestore();
        db.settings({
            timestampsInSnapshots: true
        });

        this.state = {
            username: 'castle',
            room: 'Dog Walking',
            db: db
        }

        this.selectRoom = this.selectRoom.bind(this);
        this.clearRoom = this.clearRoom.bind(this);
    }

    clearRoom() {
        this.setState({
            username: null,
            room: null
        });
    }

    selectRoom(newUsername, newRoom) {
        this.setState({
            username: newUsername,
            room: newRoom
        });
    }

    render() {
        let appInterface;
        if (this.state.room === null) {
            appInterface = (<RoomSelector selectRoom={this.selectRoom} />);
        } else {
            appInterface = (<MainInterface db={this.state.db} room={this.state.room} username={this.state.username} clearRoom={this.clearRoom} />);
        }

        return (
            <div id="app-container">
                {appInterface}
            </div>
        );
    }
}
