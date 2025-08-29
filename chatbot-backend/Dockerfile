# 1. 빌드 환경: Gradle을 사용하여 애플리케이션을 빌드합니다.
FROM gradle:8.5.0-jdk21-alpine AS build
WORKDIR /home/gradle/src

# build.gradle 파일을 먼저 복사하여 종속성을 캐싱합니다.
COPY build.gradle settings.gradle /home/gradle/src/
RUN gradle build --no-daemon > /dev/null 2>&1 || true

# 전체 소스 코드를 복사합니다.
COPY . .

# Gradle을 사용하여 실행 가능한 JAR 파일을 빌드합니다.
RUN gradle build -x test --no-daemon

# 2. 실행 환경: 빌드된 JAR 파일을 실행합니다.
FROM eclipse-temurin:21-jre
WORKDIR /app

# 빌드 환경에서 생성된 JAR 파일을 실행 환경으로 복사합니다.
COPY --from=build /home/gradle/src/build/libs/app.jar app.jar

# 컨테이너가 8080 포트를 사용한다고 외부에 알립니다.
EXPOSE 8080

# 컨테이너가 시작될 때 실행될 명령어
# 운영(prod) 프로필을 활성화합니다.
ENTRYPOINT ["java", "-jar", "-Dspring.profiles.active=prod", "app.jar"]
