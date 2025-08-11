-- V2__insert_super_admin.sql

-- 'qwer1234'를 BCrypt로 해싱한 비밀번호를 사용하세요.
-- 예시: $2a$10$abcdefg... (실제 해시값으로 교체)
INSERT INTO admins (admin_id, password, admin_name, role, created_at, updated_at)
VALUES (
           'superadmin',
           '$2a$10$QpaTtIiCfolITuXHovUUgO2EDCempsRlkMAq9PWHi1I7IqVdwuEpy',
           '슈퍼관리자',
           'SUPER_ADMIN',
           NOW(),
           NOW()
       ) ON CONFLICT (admin_id) DO NOTHING;