package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.mongodb.KnowledgeFile;
import com.incheonai.chatbotbackend.repository.mongodb.KnowledgeFileRepository;
import com.incheonai.chatbotbackend.dto.KnowledgeFileDto;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.MalformedURLException;
import java.nio.file.*;
import java.util.Date;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class KnowledgeFileService {

    private final KnowledgeFileRepository repository;

    @Value("${file.upload-dir}")
    private String uploadDir;

    /** 모든 파일 메타데이터 조회 */
    @Transactional(readOnly = true)
    public Page<KnowledgeFileDto> getFiles(Pageable pageable) {
        return repository.findAll(pageable)
                .map(KnowledgeFileDto::fromEntity);
    }

    /** 파일 저장 및 메타데이터 생성 */
    @Transactional
    public KnowledgeFileDto uploadFile(MultipartFile file, String description) {
        try {
            // 1) 물리 파일 저장
            String storedName = UUID.randomUUID() + "_" + file.getOriginalFilename();
            Path target = Paths.get(uploadDir).resolve(storedName);
            Files.createDirectories(target.getParent());
            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);

            // 2) MongoDB에 메타데이터 저장
            KnowledgeFile entity = KnowledgeFile.builder()
                    .originFilename(file.getOriginalFilename())
                    .minioObjectName(storedName)
                    .fileSize(file.getSize())
                    .fileType(file.getContentType())
                    .uploadedAt(new Date())
                    .isActive(true)
                    .description(description)
                    .build();
            KnowledgeFile saved = repository.save(entity);

            return KnowledgeFileDto.fromEntity(saved);

        } catch (IOException e) {
            throw new RuntimeException("파일 저장에 실패했습니다.", e);
        }
    }

    /** 파일 삭제 */
    @Transactional
    public void deleteFile(String fileId) {
        KnowledgeFile entity = repository.findById(fileId)
                .orElseThrow(() -> new IllegalArgumentException("삭제할 파일이 없습니다."));
        // 물리 파일 삭제
        try {
            Path target = Paths.get(uploadDir).resolve(entity.getMinioObjectName());
            Files.deleteIfExists(target);
        } catch (IOException ignored) { }
        // 메타데이터 삭제
        repository.deleteById(fileId);
    }

    /** 물리파일 로드 */
    @Transactional(readOnly = true)
    public Resource loadFileAsResource(String fileId) {
        KnowledgeFile entity = repository.findById(fileId)
                .orElseThrow(() -> new IllegalArgumentException("파일을 찾을 수 없습니다."));
        try {
            Path file = Paths.get(uploadDir)
                    .resolve(entity.getMinioObjectName())
                    .normalize();
            Resource resource = new UrlResource(file.toUri());
            if (resource.exists() && resource.isReadable()) {
                return resource;
            }
            throw new RuntimeException("읽을 수 없는 파일입니다.");
        } catch (MalformedURLException e) {
            throw new RuntimeException("파일 경로가 잘못되었습니다.", e);
        }
    }
}
