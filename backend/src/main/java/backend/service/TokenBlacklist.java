package backend.service;

import org.springframework.stereotype.Component;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class TokenBlacklist {
    private final Set<String> blacklist = ConcurrentHashMap.newKeySet();

    /** 토큰을 블랙리스트에 추가 */
    public void add(String token) {
        blacklist.add(token);
    }

    /** 블랙리스트에 있는지 확인 */
    public boolean contains(String token) {
        return blacklist.contains(token);
    }
}
