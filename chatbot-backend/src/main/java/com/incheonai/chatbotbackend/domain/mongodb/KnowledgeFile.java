package com.incheonai.chatbotbackend.domain.mongodb;

import lombok.Builder;
import lombok.Getter;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;
import java.util.Date;

@Getter
@Builder
@Document(collection = "knowledge_files") // MongoDB 컬렉션 이름 지정
public class KnowledgeFile {

    @Id
    private String id; // MongoDB의 _id 필드와 매핑

    @Field("upload_admin_id")
    private String uploadAdminId;

    @Field("minio_object_name")
    private String minioObjectName;

    @Field("origin_filename")
    private String originFilename;

    @Field("file_size")
    private Long fileSize;

    @Field("file_type")
    private String fileType;

    @Field("uploaded_at")
    private Date uploadedAt;

    @Field("is_active")
    private boolean isActive;

    @Field
    private String description;
}