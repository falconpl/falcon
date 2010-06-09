#ifndef GDK_CURSOR_HPP
#define GDK_CURSOR_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::Cursor
 */
class Cursor
    :
    public Falcon::CoreObject
{
public:

    Cursor( const Falcon::CoreClass*, const GdkCursor* = 0 );

    ~Cursor();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GdkCursor* getCursor() const { return (GdkCursor*) m_cursor; }

    void setCursor( const GdkCursor* );

    static FALCON_FUNC init( VMARG );
#if 0
    static FALCON_FUNC new_from_pixmap( VMARG );
    static FALCON_FUNC new_from_pixbuf( VMARG );
    static FALCON_FUNC new_from_name( VMARG );
    static FALCON_FUNC new_for_display( VMARG );
    static FALCON_FUNC get_display( VMARG );
    static FALCON_FUNC get_image( VMARG );
    static FALCON_FUNC ref( VMARG );
    static FALCON_FUNC unref( VMARG );
    static FALCON_FUNC destroy( VMARG );
#endif
private:

    GdkCursor*  m_cursor;

};


} // Gdk
} // Falcon

#endif // !GDK_CURSOR_HPP
