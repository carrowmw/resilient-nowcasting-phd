// Server Component — renders the full-screen digital twin UI.
// Heavy WebGL components are lazy-loaded inside MapScene (a Client Component).
import MapScene from '@/components/MapScene';

export default function Page() {
  return <MapScene />;
}
