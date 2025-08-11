-- V1__init_all_tables.sql

-- admins 테이블 생성
CREATE TABLE IF NOT EXISTS admins (
    id SERIAL PRIMARY KEY,
    admin_id VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255),
    admin_name VARCHAR(255),
    role VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_login_at TIMESTAMP,
    deleted_at TIMESTAMP
    );

-- users 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    google_id VARCHAR(255),
    password VARCHAR(255),
    login_provider VARCHAR(255) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    last_login_at TIMESTAMP,
    deleted_at TIMESTAMP
    );

-- broad_posts 테이블 생성
CREATE TABLE IF NOT EXISTS broad_posts (
    post_id BIGSERIAL PRIMARY KEY,
    author_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(255) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
    );

-- inquiries 테이블 생성
CREATE TABLE IF NOT EXISTS inquiries (
    inquiry_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(255) NOT NULL,
    answer TEXT,
    admin_id VARCHAR(255),
    urgency INTEGER,
    status VARCHAR(255) NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP
    );

-- inquiry_answers 테이블 생성
CREATE TABLE IF NOT EXISTS inquiry_answers (
    answer_id SERIAL PRIMARY KEY,
    inquiry_id INTEGER NOT NULL,
    admin_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
    );