subroutine rp1(maxm,meqn,mwaves,maux,mbc,mx,ql,qr,auxl,auxr,wave,s,amdq,apdq)

! LWR with del Castillo fundamental diagram
! returns:
! wave speed: s
! wave: wave
! left and right going fluctuation: amdq and apdq (respectively)
!
! Uses general method for flux functions that aren't necessarily strictly convex/concave
!
! CLAWPACK uses a confusing notation for left and right states in the riemann problem:
! left state: qr(1,i-1)
! right state: ql(1,i)

implicit double precision(a-h,o-z)
double precision :: wave(meqn,mwaves,1-mbc:maxm+mbc)
double precision :: ql(meqn,1-mbc:maxm+mbc)
double precision :: qr(meqn,1-mbc:maxm+mbc)
double precision :: s(mwaves,1-mbc:maxm+mbc)
double precision :: amdq(meqn,1-mbc:maxm+mbc)
double precision :: apdq(meqn,1-mbc:maxm+mbc)
common /cparam/ w, u, Z, rho_j
logical efix

efix = .true.


do 30 i=2-mbc,mx+mbc

! ------------------------------------------------
! TO DEFINE A NEW SOLVER ONLY MODIFY STUFF HERE
  ! left and right states: ul, ur
  ! flux function, derivative, and maximum: fr, fl, dflux_r, dflux_l, f0
  ul = qr(1,i-1)
  ur = ql(1,i)
  fr = Z * ( (u*ur/rho_j)**(-w) + (1.d0 - (ur/rho_j))**(-w) )**(-1.d0/w)
  fl = Z * ( (u*ul/rho_j)**(-w) + (1.d0 - (ul/rho_j))**(-w) )**(-1.d0/w)
  part1_r = ( -w*(u/rho_j) *(u*ur/rho_j)**(-w-1.d0) + (w/rho_j)*(1.d0-ur/rho_j)**(-w-1.d0) )
  part2_r = ( (u*ur/rho_j)**(-w) + (1.d0 - ur/rho_j)**(-w) )**(-1.d0/w - 1.d0)
  dflux_r = -(1.d0/w) * Z * part1_r * part2_r

  part1_l = ( -w*(u/rho_j) *(u*ul/rho_j)**(-w-1.d0) + (w/rho_j)*(1.d0-ul/rho_j)**(-w-1.d0) )
  part2_l = ( (u*ul/rho_j)**(-w) + (1.d0 - ul/rho_j)**(-w) )**(-1.d0/w - 1.d0)
  dflux_l = -(1.d0/w) * Z * part1_l * part2_l
  u_crit = rho_j / (1.d0 + u**(w/(w+1.d0)))
  f0 = Z * ( (u*u_crit/rho_j)**(-w) + (1.d0 - (u_crit/rho_j))**(-w) )**(-1.d0/w)

    ! TODO: if ul,ur > rho_j, then: fr,fl,dflux_l,dflux_r = 0.

! ------------------------------------------------
  ! wave speed
  if (ql(1,i).ne.qr(1,i-1)) then
    s(1,i) = (fr - fl) / (ur - ul)
  else
    s(1,i) = dflux_r
  endif

  ! wave: density jump
  wave(1,1,i) = ur - ul

  ! define average flux at the interface of the RP:
  if (ul.lt.ur) then
    flux_int = dmin1(fl,fr)
  else if (dflux_l.lt.0.d0 .and. dflux_r.gt.0.d0) then
    flux_int = f0
  else
    flux_int = dmax1(fl,fr)
  endif

  ! left and right going fluctuations
  amdq(1,i) = flux_int  - fl
  apdq(1,i) = fr - flux_int

30 end do
return
end subroutine rp1
