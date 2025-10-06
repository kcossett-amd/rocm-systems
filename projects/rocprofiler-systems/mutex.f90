!! Tests:
!! - ROCPROFILER_OMPT_ID_lock_init
!! - ROCPROFILER_OMPT_ID_lock_destroy
!! - ROCPROFILER_OMPT_ID_mutex_acquire
!! - ROCPROFILER_OMPT_ID_mutex_acquired
!! - ROCPROFILER_OMPT_ID_mutex_released

program mutex_test
    use omp_lib
    implicit none
    
    integer(omp_lock_kind) :: simple_lock
    integer(omp_nest_lock_kind) :: nest_lock
    integer :: thread_id
    
    ! Test lock_init and lock_destroy
    call omp_init_lock(simple_lock)
    
    !$omp parallel num_threads(4) private(thread_id)
        thread_id = omp_get_thread_num()
        
        call omp_set_lock(simple_lock)
        
        ! Do some work while holding the lock
        call sleep(1)
        
        call omp_unset_lock(simple_lock)
    !$omp end parallel
    
    call omp_destroy_lock(simple_lock)
    
end program mutex_test