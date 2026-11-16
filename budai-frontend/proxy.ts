import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function proxy(request: NextRequest) {
  const token =
    request.cookies.get('__Secure-neon-auth.session_token')?.value ||
    request.cookies.get('better-auth.session_token')?.value ||
    request.cookies.get('budai_token')?.value;
  const path = request.nextUrl.pathname;

  const isPublicPath =
    path === '/' ||
    path === '/login' ||
    path === '/register' ||
    path === '/onboarding' ||
    path === '/reset-password' ||
    path === '/callback';

  if ((path === '/login' || path === '/register') && token) {
    return NextResponse.redirect(new URL('/home', request.url));
  }

  if (!token && !isPublicPath) {
    if (
      !path.startsWith('/api/') &&
      !path.startsWith('/_next/') &&
      !path.includes('.')
    ) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
