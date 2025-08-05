package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.*;
import com.incheonai.chatbotbackend.service.BroadPostService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/broad-posts")
@RequiredArgsConstructor
public class BroadPostController {

    private final BroadPostService postService;

    /** 게시판 목록 가져오기 */
    @GetMapping
    public ResponseEntity<Page<BroadPostDto>> getPosts(
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search,
            Pageable pageable
    ) {
        return ResponseEntity.ok(postService.getPosts(category, search, pageable));
    }

    /** 단일 게시글 상세 가져오기 */
    @GetMapping("/{post_id}")
    public ResponseEntity<BroadPostDetailDto> getPost(
            @PathVariable("post_id") Long postId
    ) {
        return ResponseEntity.ok(postService.getPost(postId));
    }

    /** 게시글 작성 */
    @PostMapping
    public ResponseEntity<BroadPostDetailDto> createPost(
            @RequestBody BroadPostRequestDto req
    ) {
        return ResponseEntity.ok(postService.createPost(req));
    }

    /** 게시글 수정 */
    @PatchMapping("/{post_id}")
    public ResponseEntity<BroadPostDetailDto> updatePost(
            @PathVariable("post_id") Long postId,
            @RequestBody BroadPostRequestDto req
    ) {
        return ResponseEntity.ok(postService.updatePost(postId, req));
    }

    /** 게시글 삭제 */
    @DeleteMapping("/{post_id}")
    public ResponseEntity<Void> deletePost(
            @PathVariable("post_id") Long postId
    ) {
        postService.deletePost(postId);
        return ResponseEntity.noContent().build();
    }
}
