#ifndef G_PARAMSPEC_HPP
#define G_PARAMSPEC_HPP

#include "modgtk.hpp"


namespace Falcon {
namespace Glib {

/**
 *  \class Falcon::Glib::ParamSpec
 */
class ParamSpec
    :
    public Falcon::CoreObject
{
public:

    ParamSpec( const Falcon::CoreClass*, const GParamSpec* = 0 );

    ~ParamSpec();

    Falcon::CoreObject* clone() const { return 0; }

    bool getProperty( const Falcon::String&, Falcon::Item& ) const;

    bool setProperty( const Falcon::String&, const Falcon::Item& );

    static Falcon::CoreObject* factory( const Falcon::CoreClass*, void*, bool );

    static void modInit( Falcon::Module* );

};


} // Glib
} // Falcon

#endif // !G_PARAMSPEC_HPP
