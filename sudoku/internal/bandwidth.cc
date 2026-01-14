#include <iostream>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#include <sstream>

uintptr_t get_physical_address_once(uintptr_t virtual_addr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    size_t pagesize = getpagesize();
    off_t offset = (virtual_addr / pagesize) * sizeof(uint64_t);
    uint64_t entry;
    if (pread(fd, &entry, sizeof(entry), offset) != sizeof(entry)) {
        close(fd); 
        return 0;
    }
    close(fd);
    if (!(entry & (1ULL << 63))) return 0;
    uintptr_t pfn = entry & ((1ULL << 55) - 1);
    return (pfn * pagesize) | (virtual_addr % pagesize);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <Number of HugePages> <Mask1,Mask2,...>" 
                  << std::endl;
        return 1;
    }

    int num_pages = std::stoi(argv[1]);
    std::string mask_str = argv[2];
    std::vector<uintptr_t> masks;
    std::stringstream ss(mask_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        masks.push_back(std::stoull(item, nullptr, 16));
    }

    const size_t HUGE_PAGE_SIZE = 1024ULL * 1024ULL * 1024ULL; // 1GB
    size_t total_bytes = (size_t)num_pages * HUGE_PAGE_SIZE;

    void* ptr = mmap(NULL, total_bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << 26),
                     -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // [Important] Initialization identical to the original version
    // (touch only the first byte of each page)
    char* base_ptr = static_cast<char*>(ptr);
    for (size_t p = 0; p < (size_t)num_pages; ++p) {
        base_ptr[p * HUGE_PAGE_SIZE] = 0;
    }

    for (uintptr_t mask : masks) {
        std::vector<double*> valid_ptrs;

        // Filtering is performed internally without printing per-mask messages
        for (size_t p = 0; p < (size_t)num_pages; ++p) {
            uintptr_t page_vaddr =
                reinterpret_cast<uintptr_t>(base_ptr + (p * HUGE_PAGE_SIZE));
            uintptr_t page_paddr = get_physical_address_once(page_vaddr);
            if (page_paddr == 0) continue;

            #pragma omp parallel
            {
                std::vector<double*> local_ptrs;
                #pragma omp for nowait
                for (size_t offset = 0; offset < HUGE_PAGE_SIZE; offset += 64) {
                    uintptr_t current_paddr = page_paddr + offset;
                    if ((current_paddr & mask) == 0) {
                        local_ptrs.push_back(
                            reinterpret_cast<double*>(page_vaddr + offset));
                    }
                }
                #pragma omp critical
                valid_ptrs.insert(valid_ptrs.end(),
                                  local_ptrs.begin(),
                                  local_ptrs.end());
            }
        }

        if (valid_ptrs.empty()) continue;

        // Measurement section (identical to the original)
        double global_sum = 0.0;
        size_t n = valid_ptrs.size();
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:global_sum)
        for (size_t i = 0; i < n; ++i) {
            global_sum += *(valid_ptrs[i]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double read_bytes = n * 64.0;
        double bandwidth =
            (read_bytes / (1024.0 * 1024.0 * 1024.0)) / diff.count();

        // Preserve output format
        std::cout << "Mask: 0x" << std::hex << mask << std::dec
                  << " | Read bandwidth: " << bandwidth << " GB/s"
                  << " (Lines: " << n << ")" << std::endl;
    }

    munmap(ptr, total_bytes);
    return 0;
}

