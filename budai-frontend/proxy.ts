import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function proxy(request: NextRequest) {
  const token = request.cookies.get('budai_token')?.value;
  const path = request.nextUrl.pathname;

  const isPublicPath = path === '/' || path === '/login' || path === '/register';

  // Redirect authenticated users away from public pages
  if (isPublicPath && token) {
    return NextResponse.redirect(new URL('/home', request.url));
  }

  // Redirect unauthenticated users away from protected pages
  if (!isPublicPath && !token) {
    // Exclude Next.js internals, API routes, and static files
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
