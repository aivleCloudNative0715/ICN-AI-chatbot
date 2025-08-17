// com/incheonai/chatbotbackend/service/AirportInfoService.java
package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.dto.external.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.util.UriComponentsBuilder;
import reactor.core.publisher.Mono;
import java.net.URI;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Collections;
import java.util.List;

@Slf4j
@Service
public class AirportInfoService {

    private final WebClient webClient;

    @Value("${api.service-key}")
    private String serviceKey;
    @Value("${api.url.parking}")
    private String parkingApiUrl;
    @Value("${api.url.forecast}")
    private String forecastApiUrl;
    @Value("${api.url.weather}")
    private String weatherApiUrl;
    @Value("${api.url.flight_arrivals}")
    private String flightArrivalsApiUrl;
    @Value("${api.url.flight_departures}")
    private String flightDeparturesApiUrl;

    public AirportInfoService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
    }

    private <T> Mono<List<T>> fetchApiData(String baseUrl, MultiValueMap<String, String> queryParams, ParameterizedTypeReference<ApiResult<T>> typeReference) {
        queryParams.add("serviceKey", serviceKey);
        queryParams.add("type", "json");

        URI uri = UriComponentsBuilder.fromHttpUrl(baseUrl).queryParams(queryParams).build(false).toUri();

        return webClient.get()
                .uri(uri)
                .retrieve()
                .bodyToMono(typeReference)
                .map(result -> {
                    // 1. 최상위 'response' 객체를 먼저 가져옵니다.
                    if (result != null && result.getResponse() != null) {
                        ApiResponseData<T> responseData = result.getResponse();
                        ApiResponseData.Header header = responseData.getHeader();
                        ApiResponseData.Body<T> body = responseData.getBody();

                        // 2. 'header'와 'body'는 responseData 객체에서 가져옵니다.
                        if (header != null && "00".equals(header.getResultCode())) {
                            if (body != null && body.getItems() != null) {
                                return body.getItems();
                            }
                        } else if (header != null) {
                            log.error("API returned an error from [{}]: Code={}, Msg={}", baseUrl, header.getResultCode(), header.getResultMsg());
                        }
                    }
                    log.warn("API response from [{}] is empty or has an unexpected structure.", baseUrl);
                    return Collections.<T>emptyList();
                })
                .doOnError(error -> log.error("API call to {} failed: {}", baseUrl, error.getMessage()))
                .onErrorResume(e -> Mono.empty());
    }

    public Mono<List<ParkingInfoItem>> getParkingInfo() {
        return fetchApiData(parkingApiUrl, new LinkedMultiValueMap<>(), new ParameterizedTypeReference<>() {});
    }

    public Mono<List<PassengerForecastItem>> getPassengerForecast(String selectDate) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        if ("0".equals(selectDate) || "1".equals(selectDate)) {
            params.add("selectdate", selectDate);
        }
        return fetchApiData(forecastApiUrl, params, new ParameterizedTypeReference<>() {});
    }

    public Mono<List<ArrivalWeatherInfoItem>> getArrivalsWeatherInfo() {
        return fetchApiData(weatherApiUrl, new LinkedMultiValueMap<>(), new ParameterizedTypeReference<>() {});
    }

    public Mono<List<FlightArrivalInfoItem>> getFlightArrivalsInfo(String flightId) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();

        if (flightId == null || flightId.isEmpty()) {
            // 오늘 날짜를 "YYYYMMDD" 형식으로 만듭니다.
            String todayStr = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            String currentTime = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmm"));

            // searchday 파라미터를 추가하여 조회 날짜를 오늘로 고정합니다.
            params.add("searchday", todayStr);
            params.add("from_time", currentTime);
            params.add("to_time", "2359");
        } else {
            params.add("flight_id", flightId);
        }

        return fetchApiData(flightArrivalsApiUrl, params, new ParameterizedTypeReference<>() {});
    }

    public Mono<List<FlightDepartureInfoItem>> getFlightDeparturesInfo(String flightId) {
        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();

        if (flightId == null || flightId.isEmpty()) {
            String todayStr = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
            String currentTime = LocalTime.now().format(DateTimeFormatter.ofPattern("HHmm"));

            // searchday 파라미터를 추가하여 조회 날짜를 오늘로 고정합니다.
            params.add("searchday", todayStr);
            params.add("from_time", currentTime);
            params.add("to_time", "2359");
        } else {
            params.add("flight_id", flightId);
        }

        return fetchApiData(flightDeparturesApiUrl, params, new ParameterizedTypeReference<>() {});
    }
}