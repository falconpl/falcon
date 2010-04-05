#ifndef GDK_DRAGCONTEXT_HPP
#define GDK_DRAGCONTEXT_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Gdk {

/**
 *  \class Falcon::Gdk::DragContext
 */
class DragContext
    :
    public Falcon::CoreObject
{
public:

    DragContext( const Falcon::CoreClass*, const GdkDragContext* = 0 );

    ~DragContext();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

};


} // Gdk
} // Falcon

#endif // !GDK_DRAGCONTEXT_HPP
