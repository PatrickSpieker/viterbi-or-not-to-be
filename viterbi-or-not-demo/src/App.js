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
            username: null,
            room: null,
            db: db
        }

        this.createRoom = this.createRoom.bind(this);
        this.copyRoom = this.copyRoom.bind(this);
        this.selectRoom = this.selectRoom.bind(this);
        this.clearRoom = this.clearRoom.bind(this);
    }

    createRoom(newUsername) {
        this.state.db.collection('metadata').doc('reserved_rooms').get()
            .then(document => {
                if (document.exists) {
                    let takenCodes = document.data().reserved_rooms;
                    let code = Math.floor(Math.random() * 9000) + 1000;

                    // Ensure the code is not taken
                    while (takenCodes.includes(code.toString())) {
                        code = Math.floor(Math.random() * 9000) + 1000;   
                    }

                    code = code.toString();

                    this.state.db.collection(code).add({
                        author: newUsername,
                        timestamp: Date.now(),
                        action: 'room_created',
                    });

                    takenCodes.push(code);
                    this.state.db.collection('metadata').doc('reserved_rooms').update({
                        'reserved_rooms': takenCodes
                    });

                    this.setState({
                        username: newUsername,
                        room: code
                    });
                }
            });
    }

    copyRoom(newUsername, copyRoom) {
        this.state.db.collection('metadata').doc('reserved_rooms').get()
            .then(document => {
                if (document.exists) {
                    this.state.db.collection(copyRoom).get()
                        .then(collection => {
                            let takenCodes = document.data().reserved_rooms;
                            let code = Math.floor(Math.random() * 9000) + 1000;

                            // Ensure the code is not taken
                            while (takenCodes.includes(code.toString())) {
                                code = Math.floor(Math.random() * 9000) + 1000;   
                            }

                            code = code.toString();

                            // this.state.db.collection(code).add({
                            //     author: newUsername,
                            //     timestamp: Date.now(),
                            //     action: 'room_created'
                            // });

                            collection.forEach(doc => {
                                this.state.db.collection(code).add(doc.data());
                            });

                            takenCodes.push(code);
                            this.state.db.collection('metadata').doc('reserved_rooms').update({
                                'reserved_rooms': takenCodes
                            });

                            this.setState({
                                username: newUsername,
                                room: code
                            });
                        })
                }
            });
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
            appInterface = (<RoomSelector selectRoom={this.selectRoom} createRoom={this.createRoom} copyRoom={this.copyRoom} />);
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
