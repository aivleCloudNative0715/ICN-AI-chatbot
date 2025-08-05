package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.BroadPost;
import com.incheonai.chatbotbackend.dto.*;
import com.incheonai.chatbotbackend.repository.jpa.BroadPostRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class BroadPostService {

    private final BroadPostRepository postRepository;

    /**
     * 게시판 목록 조회
     * @param category  카테고리 필터 (optional)
     * @param search    제목/내용 검색어 (optional)
     * @param pageable  페이징 정보(page, size, sort)
     */
    @Transactional(readOnly = true)
    public Page<BroadPostDto> getPosts(
            String category, String search, Pageable pageable
    ) {
        Page<BroadPost> page;
        if (category != null && search != null) {
            page = postRepository.findByIsDeletedFalseAndCategoryAndTitleContainingOrContentContaining(
                    category, search, search, pageable
            );
        } else {
            page = postRepository.findAllByIsDeletedFalse(pageable);
        }
        return page.map(BroadPostDto::fromEntity);
    }

    /**
     * 단일 게시글 조회
     */
    @Transactional(readOnly = true)
    public BroadPostDetailDto getPost(Long postId) {
        BroadPost p = postRepository.findById(postId)
                .filter(b -> !b.getIsDeleted())
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 게시글입니다."));
        return BroadPostDetailDto.fromEntity(p);
    }

    /**
     * 게시글 등록
     */
    @Transactional
    public BroadPostDetailDto createPost(BroadPostRequestDto req) {
        BroadPost saved = postRepository.save(
                BroadPost.builder()
                        .authorId(req.authorId())
                        .title(req.title())
                        .content(req.content())
                        .category(req.category())
                        .build()
        );
        return BroadPostDetailDto.fromEntity(saved);
    }

    /**
     * 게시글 수정
     */
    @Transactional
    public BroadPostDetailDto updatePost(Long postId, BroadPostRequestDto req) {
        BroadPost p = postRepository.findById(postId)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 게시글입니다."));
        p.update(req.title(), req.content(), req.category());
        BroadPost updated = postRepository.save(p);
        return BroadPostDetailDto.fromEntity(updated);
    }

    /**
     * 게시글 삭제 (soft delete)
     */
    @Transactional
    public void deletePost(Long postId) {
        BroadPost p = postRepository.findById(postId)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 게시글입니다."));
        p.delete();
        postRepository.save(p);
    }
}
