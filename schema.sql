DROP TABLE IF EXISTS Bins;
DROP TABLE IF EXISTS Points;
DROP TABLE IF EXISTS Timeseries;
DROP TABLE IF EXISTS Msus;
DROP TABLE IF EXISTS MsuTypes;
DROP TABLE IF EXISTS Threads;
DROP TABLE IF EXISTS Runtimes;
DROP TABLE IF EXISTS Statistics;

CREATE TABLE Statistics (
    id int NOT NULL,
    name varchar(64),
    monotonic TINYINT,
    PRIMARY KEY (id)
);

CREATE TABLE Runtimes (
    id  int NOT NULL,
    PRIMARY KEY(id)
);

CREATE TABLE Threads (
    pk  int NOT NULL AUTO_INCREMENT,
    thread_id int NOT NULL,
    runtime_id int NOT NULL,
    FOREIGN KEY (runtime_id) REFERENCES Runtimes(id),
    INDEX rt_th_idx (runtime_id, thread_id),
    PRIMARY KEY (pk)
);

CREATE TABLE MsuTypes (
    id  int NOT NULL,
    name varchar(64) NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE Msus (
    pk  int NOT NULL AUTO_INCREMENT,
    msu_id int NOT NULL,
    msu_type_id int NOT NULL,
    thread_pk int NOT NULL,
    FOREIGN KEY (msu_type_id) REFERENCES MsuTypes(id),
    FOREIGN KEY (thread_pk) REFERENCES Threads(pk),
    INDEX id_index (msu_id),
    PRIMARY KEY (pk)
);

CREATE TABLE Timeseries (
    pk int NOT NULL AUTO_INCREMENT,
    runtime_id int,
    thread_pk int,
    msu_pk int,
    statistic_id int NOT NULL,
    max_limit DECIMAL(18, 9),
    FOREIGN KEY (runtime_id) REFERENCES Runtimes(id),
    FOREIGN KEY (thread_pk) REFERENCES Threads(pk),
    FOREIGN KEY (msu_pk) REFERENCES Msus(pk),
    FOREIGN KEY (statistic_id) REFERENCES Statistics(id),
    INDEX runtime_idx (runtime_id, statistic_id),
    INDEX thread_idx (thread_pk, statistic_id),
    INDEX msu_idx (msu_pk, statistic_id),
    PRIMARY KEY (pk)
);

CREATE TABLE Points (
    pk int NOT NULL AUTO_INCREMENT,
    ts DECIMAL(19, 9) NOT NULL,
    timeseries_pk int NOT NULL,
    size int NOT NULL,
    mean DECIMAL(18, 9),
    std DECIMAL(18, 9),
    INDEX time_index (ts),
    FOREIGN KEY (timeseries_pk) REFERENCES Timeseries(pk),
    CONSTRAINT unique_ts UNIQUE (ts, timeseries_pk),
    PRIMARY KEY (pk)
);

CREATE TABLE Bins (
    pk int NOT NULL AUTO_INCREMENT,
    value DECIMAL(18, 9) NOT NULL,
    percentile DECIMAL(5, 2) NOT NULL,
    points_pk int NOT NULL,
    FOREIGN KEY (points_pk) REFERENCES Points(pk),
    CONSTRAINT unique_pctile UNIQUE (pk, percentile),
    PRIMARY KEY (pk)
);
