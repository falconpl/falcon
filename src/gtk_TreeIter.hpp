#ifndef GTK_TREEITER_HPP
#define GTK_TREEITER_HPP

#include "modgtk.hpp"

#define GET_TREEITER( item ) \
        (Falcon::dyncast<Gtk::TreeIter*>( (item).asObjectSafe() )->getTreeIter())


namespace Falcon {
namespace Gtk {

/**
 *  \class Falcon::Gtk::TreeIter
 */
class TreeIter
    :
    public Falcon::CoreObject
{
public:

    TreeIter( const Falcon::CoreClass*, const GtkTreeIter* = 0 );

    ~TreeIter();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

    GtkTreeIter* getTreeIter() const { return (GtkTreeIter*) &m_iter; }

    static FALCON_FUNC copy( VMARG );

private:

    GtkTreeIter     m_iter;

};


} // Gtk
} // Falcon

#endif // !GTK_TREEITER_HPP
