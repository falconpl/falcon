#ifndef GDK_PIXMAP_HPP
#define GDK_PIXMAP_HPP

#include "modgtk.hpp"

#define GET_PIXMAP( item ) \
        ((GdkPixmap*) Falcon::dyncast<Gdk::Pixmap*>( (item).asObjectsafe() )->getGObject())


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Pixmap
 */
class Pixmap
    :
    public Gtk::CoreGObject
{
public:

    Pixmap( const Falcon::CoreClass*, const GdkPixmap* = 0 );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    static FALCON_FUNC init( VMARG );

    static FALCON_FUNC create_from_data( VMARG );
#if 0 // todo
    static FALCON_FUNC create_from_xpm( VMARG );

    static FALCON_FUNC colormap_create_from_xpm( VMARG );

    static FALCON_FUNC create_from_xpm_d( VMARG );

    static FALCON_FUNC colormap_create_from_xpm_d( VMARG );
#endif
};


} // Gdk
} // Falcon

#endif // !GDK_PIXMAP_HPP
